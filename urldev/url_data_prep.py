"""
url_data_prep.py
Merges phishing_site_urls.csv (good/bad) and PhishTank_2026.csv (all phishing)
into one dataset, then engineers standard phishing-URL lexical features.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# EDIT THIS to wherever you put the two URL CSVs
DATA_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\urldev\urls"

# EDIT THIS to where you want the merged dataset saved
OUT_PATH = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\Major Project\Dev\urldev\merged_url_dataset.csv"

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "signin", "banking",
    "confirm", "webscr", "ebayisapi", "paypal", "password", "suspend",
]


def ensure_scheme(url):
    if not re.match(r"^[a-zA-Z]+://", url):
        return "http://" + url
    return url


def extract_features(url):
    raw = str(url)
    try:
        parsed = urlparse(ensure_scheme(raw))
        host = parsed.netloc or ""
        path = parsed.path or ""
    except ValueError:
        # malformed URL (e.g. stray brackets parsed as invalid IPv6) - fall back gracefully
        host = ""
        path = raw

    # Strip scheme before deriving text/lexical features. PhishTank_2026 always
    # includes "https://" while phishing_site_urls.csv never includes any scheme -
    # so scheme presence is a source artifact, not a genuine phishing signal, and
    # would otherwise leak into both the char n-grams and any has_https feature.
    stripped = re.sub(r"^[a-zA-Z]+://", "", raw)

    features = {}
    features["url_length"] = len(stripped)
    features["num_dots"] = stripped.count(".")
    features["num_hyphens"] = stripped.count("-")
    features["num_at"] = stripped.count("@")
    features["num_digits"] = sum(c.isdigit() for c in stripped)
    features["num_subdirs"] = path.count("/")
    features["num_params"] = stripped.count("=") + stripped.count("&")
    features["has_ip"] = int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}", host)))
    features["num_subdomains"] = max(host.count(".") - 1, 0) if host else 0
    features["is_shortened"] = int(bool(re.search(
        r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly", stripped, re.I
    )))
    features["suspicious_word_count"] = sum(
        1 for w in SUSPICIOUS_WORDS if w in stripped.lower()
    )
    features["unique_char_ratio"] = len(set(stripped)) / max(len(stripped), 1)
    features["url_stripped"] = stripped  # used as TF-IDF input, scheme-free
    return features


def build_dataset(sample_per_class=None):
    print("Loading phishing_site_urls.csv ...")
    df1 = pd.read_csv(
        f"{DATA_DIR}\\phishing_site_urls.csv",
        dtype=str, on_bad_lines="skip"
    )
    df1 = df1.dropna(subset=["URL", "Label"])
    df1["label"] = (df1["Label"].str.strip().str.lower() == "bad").astype(int)
    df1 = df1[["URL", "label"]].rename(columns={"URL": "url"})

    print("Loading PhishTank_2026.csv ...")
    df2 = pd.read_csv(
        f"{DATA_DIR}\\PhishTank_2026.csv",
        dtype=str, on_bad_lines="skip"
    )
    df2 = df2.dropna(subset=["URL", "Label"])
    df2["label"] = 1  # PhishTank is all confirmed phishing
    df2 = df2[["URL", "label"]].rename(columns={"URL": "url"})

    full = pd.concat([df1, df2], ignore_index=True)
    full = full.drop_duplicates(subset="url").reset_index(drop=True)

    print(f"Combined + deduped: {len(full)} rows")
    print(full["label"].value_counts())

    # Optional balanced downsample - leave as None on your machine to use ALL data
    if sample_per_class:
        legit = full[full.label == 0]
        phish = full[full.label == 1]
        n = min(sample_per_class, len(legit), len(phish))
        full = pd.concat([
            legit.sample(n, random_state=42),
            phish.sample(n, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Downsampled to balanced {len(full)} rows ({n} per class)")

    print("Extracting URL features (this can take a few minutes on full data)...")
    feat_rows = full["url"].apply(extract_features).apply(pd.Series)
    full = pd.concat([full, feat_rows], axis=1)

    return full


if __name__ == "__main__":
    # Use the FULL dataset on your machine (no downsampling needed with more RAM/CPU)
    df = build_dataset(sample_per_class=None)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved merged dataset to {OUT_PATH}")