"""
train_sms_model_distilbert.py
Trains the SMS spam/phishing detection model for SecureShield using
DistilBERT fine-tuning. Feasible on CPU since the dataset is small
(~5,572 rows) - unlike email (~203K rows), which used XGBoost instead.

Outputs saved to model_artifacts/:
  - pytorch_model.bin / model.safetensors + config.json (the fine-tuned model)
  - tokenizer files
  - metrics.json
"""

import pandas as pd
import numpy as np
import os
import json

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# EDIT THESE to match your folder
DATA_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\smsdev\merged_sms_dataset.csv"
OUT_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\smsdev\model_artifacts_distilbert"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    # SMS messages are short - 64 tokens comfortably covers almost all of them,
    # unlike email's 256, which keeps CPU training fast.
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )


train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    probs_positive = np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1)  # softmax prob of class 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs_positive)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=16,   # small dataset -> can afford a bigger batch than email
    per_device_eval_batch_size=16,
    num_train_epochs=4,               # small dataset -> a few more epochs is cheap and helps
    learning_rate=2e-5,
    eval_strategy="epoch",            # use evaluation_strategy="epoch" instead on transformers <4.41
    save_strategy="epoch",
    save_total_limit=1,               # keep only the best/last checkpoint - avoids filling disk
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("Training DistilBERT on SMS data (should take roughly 15-30 min on CPU)...")
trainer.train()

metrics = trainer.evaluate()
print("\n=== FINAL METRICS ===")
print(json.dumps(metrics, indent=2))

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSMS DistilBERT model saved to {OUT_DIR}")