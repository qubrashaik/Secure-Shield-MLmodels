"""
train_url_model.py
Trains the URL phishing detection model for SecureShield.
Character n-gram TF-IDF + engineered lexical features -> XGBClassifier.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

# EDIT THESE to match your folder
DATA_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\urldev\merged_url_dataset.csv"
OUT_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\urldev\model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

ENGINEERED_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_at", "num_digits",
    "num_subdirs", "num_params", "has_ip", "num_subdomains",
    "is_shortened", "suspicious_word_count", "unique_char_ratio",
]


def main():
    print("Loading merged URL dataset...")
    df = pd.read_csv(DATA_PATH)
    df["url_stripped"] = df["url_stripped"].fillna("")

    X_text = df["url_stripped"]
    X_engineered = df[ENGINEERED_COLS].values
    y = df["label"].values

    (X_text_train, X_text_test,
     X_eng_train, X_eng_test,
     y_train, y_test) = train_test_split(
        X_text, X_engineered, y,
        test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_text_train)} | Test size: {len(X_text_test)}")

    print("Fitting character n-gram TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10000,
        min_df=5,
    )
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)

    X_train_full = hstack([X_tfidf_train, csr_matrix(X_eng_train)]).tocsr()
    X_test_full = hstack([X_tfidf_test, csr_matrix(X_eng_test)]).tocsr()

    print("Training XGBoostClassifier...")
    start = time.time()
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=neg / pos,
    )
    model.fit(X_train_full, y_train)
    print(f"Training took {time.time() - start:.1f}s")

    y_pred = model.predict(X_test_full)
    y_proba = model.predict_proba(X_test_full)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    feature_names = list(vectorizer.get_feature_names_out()) + ENGINEERED_COLS
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop 15 most important features:")
    print(fi_df.head(15).to_string(index=False))

    joblib.dump(vectorizer, os.path.join(OUT_DIR, "char_tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(OUT_DIR, "url_model.pkl"))
    fi_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()