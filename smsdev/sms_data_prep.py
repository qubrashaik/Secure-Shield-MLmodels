"""
sms_data_prep.py
Loads the SMSSpamCollection dataset, cleans it, and engineers
SMS-specific features (short-text spam patterns differ from email).
"""

import pandas as pd
import re

# EDIT THIS to wherever you put the SMSSpamCollection file, e.g.:
# r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\smsdev\SMSSpamCollection"
RAW_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\smsdev\sms\SMSSpamCollection"

# EDIT THIS to where you want the merged dataset saved
OUT_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\smsdev\merged_sms_dataset.csv"


def build_dataset():
    df = pd.read_csv(
        RAW_PATH, sep="\t", header=None, names=["label_str", "text"],
        encoding="latin-1"
    )

    df["label"] = (df["label_str"] == "spam").astype(int)
    df["text"] = df["text"].fillna("").astype(str)

    # SMS-specific engineered features
    df["text_length"] = df["text"].str.len()
    df["has_url"] = df["text"].str.contains(
        r"https?://|www\.|\bbit\.ly\b", case=False, regex=True, na=False
    ).astype(int)
    df["digit_count"] = df["text"].str.count(r"\d")
    df["has_phone_number"] = df["text"].str.contains(
        r"\b\d{5,}\b", regex=True, na=False
    ).astype(int)
    df["has_currency"] = df["text"].str.contains(
        r"[£$€]|\bfree\b|\bwin\b|\bprize\b|\bcash\b", case=False, regex=True, na=False
    ).astype(int)
    df["exclaim_count"] = df["text"].str.count("!")
    df["uppercase_ratio"] = df["text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["urgency_flag"] = df["text"].str.contains(
        r"urgent|call now|claim|winner|congratulations|verify|act now|limited",
        case=False, regex=True, na=False
    ).astype(int)

    print(f"Total rows: {len(df)}")
    print(df["label"].value_counts())

    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved merged dataset to {OUT_PATH}")