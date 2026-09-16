"""
data_prep.py
Loads all 7 email datasets, standardizes schema, cleans labels,
and merges into a single dataframe: subject, body, has_url, label
"""

import pandas as pd
import numpy as np
import re
import os

DATA_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\emaildataset"

FILES = [
    "Assassin.csv",
    "CEAS-08.csv",
    "Enron.csv",
    "Ling.csv",
    "TREC-05.csv",
    "TREC-06.csv",
    "TREC-07.csv",
]


def clean_label_column(df):
    """Keep only rows where label is cleanly 0 or 1 (drops malformed parse rows)."""
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0.0, 1.0])]
    df["label"] = df["label"].astype(int)
    return df


def load_and_standardize(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    # Standardize columns: keep subject, body, label, and has_url if available
    if "urls" in df.columns:
        has_url = pd.to_numeric(df["urls"], errors="coerce").fillna(0).astype(int)
        has_url = (has_url > 0).astype(int)
    else:
        has_url = None

    df = clean_label_column(df)

    out = pd.DataFrame()
    out["subject"] = df.get("subject", "").fillna("")
    out["body"] = df.get("body", "").fillna("")
    out["label"] = df["label"]

    if has_url is not None:
        out["has_url"] = has_url.reindex(df.index).fillna(0).astype(int)
    else:
        # infer from body text if urls column wasn't present
        out["has_url"] = out["body"].str.contains(
            r"https?://|www\.", case=False, regex=True, na=False
        ).astype(int)

    out["source"] = filename.replace(".csv", "")
    return out


def build_dataset():
    frames = []
    for f in FILES:
        print(f"Loading {f} ...")
        frames.append(load_and_standardize(f))

    full = pd.concat(frames, ignore_index=True)

    # combine subject + body into one text field
    full["text"] = (full["subject"].astype(str) + " " + full["body"].astype(str)).str.strip()

    # drop empty text rows
    full = full[full["text"].str.len() > 0].reset_index(drop=True)

    # basic engineered features
    full["text_length"] = full["text"].str.len()
    full["exclaim_count"] = full["text"].str.count("!")
    full["urgency_flag"] = full["text"].str.contains(
        r"\b(urgent|verify|immediately|suspend|password|click here|act now|limited time|winner|congratulations)\b",
        case=False, regex=True, na=False
    ).astype(int)

    print(f"\nFinal merged dataset: {len(full)} rows")
    print(full["label"].value_counts())
    print(f"Sources: {full['source'].value_counts().to_dict()}")

    return full


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\emaildev\merged_email_dataset.csv", index=False)
    print("\nSaved merged dataset to merged_email_dataset.csv")
