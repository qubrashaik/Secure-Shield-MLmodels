# """
# train_email_model.py
# Trains the email phishing/spam detection model for SecureShield.

# Pipeline:
#   TF-IDF (text)  ----+
#                       |--> hstack --> XGBClassifier --> label + confidence
#   engineered feats ---+

# Outputs saved to model_artifacts/:
#   - tfidf_vectorizer.pkl
#   - email_model.pkl
#   - metrics.json
#   - feature_importances.csv
# """

# import pandas as pd
# import numpy as np
# import json
# import os
# import joblib
# import time

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_auc_score, confusion_matrix, classification_report
# )
# from scipy.sparse import hstack, csr_matrix
# from xgboost import XGBClassifier

# OUT_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\model_artifacts"
# os.makedirs(OUT_DIR, exist_ok=True)

# RANDOM_STATE = 42


# def main():
#     print("Loading merged dataset...")
#     df = pd.read_csv(r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\merged_email_dataset.csv")
#     df["text"] = df["text"].fillna("")

#     X_text = df["text"]
#     X_engineered = df[["has_url", "text_length", "exclaim_count", "urgency_flag"]].values
#     y = df["label"].values

#     # ---- Train/test split (stratified) ----
#     (X_text_train, X_text_test,
#      X_eng_train, X_eng_test,
#      y_train, y_test) = train_test_split(
#         X_text, X_engineered, y,
#         test_size=0.2, stratify=y, random_state=RANDOM_STATE
#     )

#     print(f"Train size: {len(X_text_train)} | Test size: {len(X_text_test)}")

#     # ---- TF-IDF ----
#     print("Fitting TF-IDF vectorizer...")
#     vectorizer = TfidfVectorizer(
#         max_features=8000,
#         stop_words="english",
#         ngram_range=(1, 2),
#         min_df=3,
#     )
#     X_tfidf_train = vectorizer.fit_transform(X_text_train)
#     X_tfidf_test = vectorizer.transform(X_text_test)

#     # ---- Combine TF-IDF + engineered numeric features ----
#     X_train_full = hstack([X_tfidf_train, csr_matrix(X_eng_train)]).tocsr()
#     X_test_full = hstack([X_tfidf_test, csr_matrix(X_eng_test)]).tocsr()

#     # ---- Train model (XGBoost) ----
#     print("Training XGBoostClassifier...")
#     start = time.time()
#     model = XGBClassifier(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.1,
#         eval_metric="logloss",
#         random_state=RANDOM_STATE,
#         n_jobs=-1,
#     )
#     model.fit(X_train_full, y_train)
#     print(f"Training took {time.time() - start:.1f}s")

#     # ---- Evaluate ----
#     y_pred = model.predict(X_test_full)
#     y_proba = model.predict_proba(X_test_full)[:, 1]

#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall": recall_score(y_test, y_pred),
#         "f1_score": f1_score(y_test, y_pred),
#         "roc_auc": roc_auc_score(y_test, y_proba),
#         "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
#         "train_size": len(y_train),
#         "test_size": len(y_test),
#     }

#     print("\n=== METRICS ===")
#     for k, v in metrics.items():
#         if k != "confusion_matrix":
#             print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
#     print("Confusion matrix [[TN, FP], [FN, TP]]:")
#     print(np.array(metrics["confusion_matrix"]))
#     print("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Spam/Phishing"]))

#     # ---- Feature importances ----
#     feature_names = list(vectorizer.get_feature_names_out()) + \
#         ["has_url", "text_length", "exclaim_count", "urgency_flag"]
#     importances = model.feature_importances_
#     fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
#     fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

#     print("\nTop 15 most important features:")
#     print(fi_df.head(15).to_string(index=False))

#     # ---- Save artifacts ----
#     joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))
#     joblib.dump(model, os.path.join(OUT_DIR, "email_model.pkl"))
#     fi_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
#     with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)

#     print(f"\nSaved model artifacts to {OUT_DIR}/")


# if __name__ == "__main__":
#     main()









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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\merged_email_dataset.csv"
OUT_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\model_artifacts"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_ds = Dataset.from_pandas(train_df[["text","label"]])
test_ds = Dataset.from_pandas(test_df[["text","label"]])

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
test_ds.set_format("torch", columns=["input_ids","attention_mask","label"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

with open(os.path.join(OUT_DIR,"metrics.json"),"w") as f:
    json.dump(metrics,f,indent=2)

print("Email DistilBERT model saved.")