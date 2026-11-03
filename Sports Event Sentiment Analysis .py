import os
import re
import json
import random
import warnings
import subprocess
import sys

packages = [
    "transformers",
    "torch",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "nltk",
    "tqdm",
    "sentencepiece",
    "lime",
    "shap",
    "accelerate",
    "openpyxl"
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import shap

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.utils.class_weight import compute_class_weight
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "microsoft/deberta-v3-base"
DATA_PATH = "Ghana-FIFA2022-WorldCup-Dataset.csv"
OUTPUT_DIR = "DeBiFormer_Results"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
HIDDEN_DIM = 256
NUM_HEADS = 4
DROPOUT = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset file not found: {DATA_PATH}. Update DATA_PATH to the location of your CSV file."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns.tolist())
print(df.info())
print(df.isnull().sum())
print("Duplicate records:", df.duplicated().sum())
print(df.describe(include="all"))

text_candidates = ["tweet", "text", "tweets", "content", "clean_text"]
label_candidates = ["sentiment", "label", "class", "polarity"]

TEXT_COLUMN = next(
    (col for col in text_candidates if col in df.columns),
    None
)

LABEL_COLUMN = next(
    (col for col in label_candidates if col in df.columns),
    None
)

if TEXT_COLUMN is None or LABEL_COLUMN is None:
    raise ValueError(
        "Could not automatically identify the tweet and sentiment columns. "
        "Set TEXT_COLUMN and LABEL_COLUMN manually in the script."
    )

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
df["clean_tweet"] = df[TEXT_COLUMN].apply(clean_tweet)
df = df[df["clean_tweet"].str.len() > 0].copy()

label_encoder = LabelEncoder()
df["sentiment_encoded"] = label_encoder.fit_transform(
    df[LABEL_COLUMN].astype(str)
)

CLASS_NAMES = label_encoder.classes_.tolist()
NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES < 2:
    raise ValueError("At least two sentiment classes are required.")

print("Sentiment classes:", CLASS_NAMES)
print("Number of classes:", NUM_CLASSES)
print(df[["clean_tweet", LABEL_COLUMN, "sentiment_encoded"]].head())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=LABEL_COLUMN, order=CLASS_NAMES)
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"), dpi=300)
plt.show()

df["tweet_length"] = df["clean_tweet"].apply(lambda x: len(x.split()))

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="tweet_length", hue=LABEL_COLUMN, bins=40, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Words")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tweet_length_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=LABEL_COLUMN, y="tweet_length")
plt.title("Tweet Length by Sentiment")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tweet_length_by_sentiment.png"), dpi=300)
plt.show()

print("Total Tweets:", len(df))
print("Unique Tweets:", df["clean_tweet"].nunique())
print("Average Tweet Length:", df["tweet_length"].mean())
print(df[LABEL_COLUMN].value_counts())

X = df["clean_tweet"].values
y = df["sentiment_encoded"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_encoding = tokenizer(
    X_train[0],
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print("Input IDs Shape:", sample_encoding["input_ids"].shape)
print("Attention Mask Shape:", sample_encoding["attention_mask"].shape)

class FIFAWorldCupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = FIFAWorldCupDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = FIFAWorldCupDataset(X_val, y_val, tokenizer, MAX_LENGTH)
test_dataset = FIFAWorldCupDataset(X_test, y_test, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class DeBiFormer(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        hidden_dim=256,
        num_heads=4,
        dropout=0.3
    ):
        super().__init__()

        self.deberta = AutoModel.from_pretrained(model_name)
        deberta_dim = self.deberta.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=deberta_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_attention=False):
        deberta_output = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = deberta_output.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)

        attention_output, attention_weights = self.multihead_attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True,
            average_attn_weights=False
        )

        expanded_mask = attention_mask.unsqueeze(-1).float()

        pooled_output = (
            attention_output * expanded_mask
        ).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(self.dropout(pooled_output))

        if return_attention:
            return logits, attention_weights

        return logits

model = DeBiFormer(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
).to(DEVICE)

print(model)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_predictions = torch.argmax(logits, dim=1)

        predictions.extend(batch_predictions.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(actual_labels, predictions)
    )

def validate_one_epoch(model, loader):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(actual_labels, predictions)
    )

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_model_path = os.path.join(OUTPUT_DIR, "best_debiformer.pth")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler
    )

    val_loss, val_accuracy = validate_one_epoch(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
print("Training completed.")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_validation_curves.png"), dpi=300)
plt.show()

model.eval()
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-Score:", f1)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as file:
    file.write(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title("DeBiFormer Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.show()

y_true_binary = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

plt.figure(figsize=(9, 7))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("Multiclass ROC Curves - DeBiFormer")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multiclass_roc_curves.png"), dpi=300)
plt.show()

def visualize_attention(text, model, tokenizer):
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, attention_weights = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention=True
        )

    valid_length = int(attention_mask[0].sum().item())

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0, :valid_length].cpu().tolist()
    )

    attention_weights = attention_weights[
        0, :, :valid_length, :valid_length
    ].cpu().numpy()

    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 6))

    if num_heads == 1:
        axes = [axes]

    for head in range(num_heads):
        sns.heatmap(
            attention_weights[head],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=axes[head]
        )
        axes[head].set_title(f"Attention Head {head + 1}")
        axes[head].tick_params(axis="x", rotation=90)
        axes[head].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "multihead_attention.png"), dpi=300)
    plt.show()

    predicted_class = torch.argmax(logits, dim=1).item()
    print("Predicted Sentiment:", CLASS_NAMES[predicted_class])

visualize_attention(X_test[0], model, tokenizer)

def predict_proba(texts):
    model.eval()
    results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = list(texts[i:i + BATCH_SIZE])

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probabilities = torch.softmax(logits, dim=1)

        results.extend(probabilities.cpu().numpy())

    return np.array(results)

explanation_samples = list(X_test[:20])
explanation_probabilities = predict_proba(explanation_samples)

print("Explainability samples prepared.")
print("Probability shape:", explanation_probabilities.shape)

lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    random_state=SEED
)

lime_sample = X_test[0]

lime_exp = lime_explainer.explain_instance(
    lime_sample,
    predict_proba,
    num_features=15,
    num_samples=1000
)

lime_features = lime_exp.as_list()
lime_words = [item[0] for item in lime_features]
lime_scores = [item[1] for item in lime_features]

lime_prediction = int(np.argmax(predict_proba([lime_sample])[0]))

print("Actual Tweet:", lime_sample)
print("Predicted Sentiment:", CLASS_NAMES[lime_prediction])

plt.figure(figsize=(10, 6))
lime_colors = ["green" if score > 0 else "red" for score in lime_scores]
plt.barh(lime_words, lime_scores, color=lime_colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("LIME Local Feature Contributions")
plt.xlabel("Contribution to Predicted Class")
plt.ylabel("Words")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lime_feature_contributions.png"), dpi=300)
plt.show()

shap.initjs()

shap_masker = shap.maskers.Text(tokenizer)

shap_explainer = shap.Explainer(
    predict_proba,
    shap_masker,
    output_names=CLASS_NAMES
)

shap_samples = list(X_test[:5])
shap_values = shap_explainer(shap_samples, max_evals=300)

shap.plots.text(shap_values[0, :, :])
plt.savefig(os.path.join(OUTPUT_DIR, "shap_local_explanation.png"), dpi=300, bbox_inches="tight")
plt.show()

shap.plots.bar(shap_values.abs.mean(0), max_display=15)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_feature_contributions.png"), dpi=300, bbox_inches="tight")
plt.show()

prediction_df = pd.DataFrame({
    "tweet": X_test,
    "actual_sentiment": label_encoder.inverse_transform(y_true),
    "predicted_sentiment": label_encoder.inverse_transform(y_pred)
})

for i, class_name in enumerate(CLASS_NAMES):
    prediction_df[f"probability_{class_name}"] = y_prob[:, i]

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

prediction_df.to_excel(
    os.path.join(OUTPUT_DIR, "test_predictions.xlsx"),
    index=False
)

torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "debiformer_model.pth")
)

tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

experiment_config = {
    "model": "DeBiFormer",
    "base_model": MODEL_NAME,
    "architecture": {
        "components": [
            "DeBERTa",
            "BiLSTM",
            "MultiHeadAttention",
            "FullyConnectedClassifier"
        ],
        "hidden_dimension": HIDDEN_DIM,
        "attention_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_length": MAX_LENGTH,
        "optimizer": "AdamW",
        "scheduler": "Linear Warmup Scheduler",
        "loss_function": "Weighted Cross Entropy"
    },
    "dataset": {
        "total_samples": int(len(df)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "testing_samples": int(len(X_test))
    },
    "evaluation": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "random_seed": SEED,
    "device": str(DEVICE)
}

with open(
    os.path.join(OUTPUT_DIR, "experiment_config.json"),
    "w"
) as file:
    json.dump(experiment_config, file, indent=4)

pd.DataFrame(history).to_csv(
    os.path.join(OUTPUT_DIR, "training_history.csv"),
    index=False
)

print("All experimental results saved in:", OUTPUT_DIR)
