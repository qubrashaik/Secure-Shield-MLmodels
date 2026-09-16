"""
app.py
SecureShield prediction microservice.

Loads the 3 trained models (email, sms, url) and exposes REST endpoints
for Spring Boot to call:

    POST /predict/email   { "subject": "...", "body": "..." }
    POST /predict/sms     { "text": "..." }
    POST /predict/url     { "url": "..." }

Each returns:
    {
      "label": "Legitimate" | "Spam" | "Phishing",
      "confidence": 0.94,
      "top_features": [["feature_name", importance_or_value], ...]
    }

SMS additionally returns richer XAI:
    "xai": {
        "lime": [["word", weight], ...],
        "shap": [["word", weight], ...]
    }

Run:
    pip install flask joblib pandas numpy scipy scikit-learn xgboost torch transformers lime shap
    python app.py
Then it listens on http://localhost:5000
"""

import re
import numpy as np
import joblib
import torch
from flask import Flask, request, jsonify
from scipy.sparse import hstack, csr_matrix
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap

app = Flask(__name__)

# ============================================================
# EDIT THESE PATHS to match your machine
# ============================================================
EMAIL_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\secureShield\MLmodels\emaildev\model_artifacts"

SMS_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\secureShield\MLmodels\smsdev\model_artifacts"

SMS_DISTILBERT_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\secureShield\MLmodels\smsdev\model_artifacts_distilbert"

URL_DIR = r"C:\Users\SHAIK QUBRA\OneDrive\Desktop\secureShield\MLmodels\urldev\model_artifacts"
print("Loading models...")
email_vectorizer = joblib.load(f"{EMAIL_DIR}\\tfidf_vectorizer.pkl")
email_model = joblib.load(f"{EMAIL_DIR}\\email_model.pkl")

# SMS now runs on the fine-tuned DistilBERT model instead of XGBoost
sms_tokenizer = AutoTokenizer.from_pretrained(SMS_DISTILBERT_DIR)
sms_model = AutoModelForSequenceClassification.from_pretrained(SMS_DISTILBERT_DIR)
sms_model.eval()

url_vectorizer = joblib.load(f"{URL_DIR}\\char_tfidf_vectorizer.pkl")
url_model = joblib.load(f"{URL_DIR}\\url_model.pkl")
print("All models loaded.")

# Label convention kept consistent with the other two models: 0 = Legitimate, 1 = Spam/Phishing
SMS_CLASS_NAMES = ["Legitimate", "Spam/Phishing"]


def sms_predict_proba(texts):
    """Wraps the DistilBERT SMS model so LIME/SHAP can call it like a sklearn model."""
    inputs = sms_tokenizer(
        list(texts), return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = sms_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()
    return probs


# Build explainers once at startup - NOT inside the route (SHAP init especially is expensive)
lime_sms_explainer = LimeTextExplainer(class_names=SMS_CLASS_NAMES)
shap_sms_masker = shap.maskers.Text(sms_tokenizer)
shap_sms_explainer = shap.Explainer(sms_predict_proba, shap_sms_masker)


def explain_sms_lime(text_input, pred_class, n=6):
    """Fast, always-on explanation for the live /predict/sms response. Returns
    word-level attributions with a consistent sign convention: positive weight
    pushes toward Spam/Phishing, negative pushes toward Legitimate.

    IMPORTANT: `pred_class` is the class the model actually predicted for this
    message (0 = Legitimate, 1 = Spam/Phishing). We always ask LIME to explain
    THAT class, not a hardcoded one - otherwise the explanation answers a
    different question than "why did the model decide this?" and can
    contradict the returned label (e.g. showing red/spam-leaning words for a
    message the model classified as Legitimate).

    LIME's as_list(label=k) returns weights *for class k*. To keep the
    "positive = spam-push" convention stable regardless of which class won,
    we flip the sign when explaining class 0.

    num_samples is capped - LIME's default is 5000, meaning 5000 forward
    passes through DistilBERT per explanation. That, not SHAP, was the real
    source of the slowness on a CPU-only machine. 200 is plenty for a short
    SMS message and should return in well under a second."""
    try:
        lime_exp = lime_sms_explainer.explain_instance(
            text_input, sms_predict_proba, num_features=n, labels=(pred_class,), num_samples=200
        )
        raw = lime_exp.as_list(label=pred_class)
        if pred_class == 0:
            # Explaining "Legitimate" directly returns positive = supports
            # Legitimate. Flip so positive always means "pushes toward spam",
            # matching the convention the frontend (ExplainableAIPanel) expects.
            raw = [[w, -v] for w, v in raw]
        return [[w, round(float(v), 4)] for w, v in raw]
    except Exception:
        return []


def explain_sms_shap(text_input, n=6, max_evals=64):
    """Slower, on-demand explanation - call this only from /explain/sms/shap
    when the user asks for a deeper explanation, NOT on every scan. SHAP does
    many forward passes through DistilBERT per call, which is why it doesn't
    belong on the live prediction path on a CPU-only machine."""
    shap_values = shap_sms_explainer([text_input], max_evals=max_evals)
    tokens = shap_values.data[0]
    contributions = shap_values.values[0][:, 1]  # contribution toward Spam/Phishing class
    pairs = list(zip(tokens, contributions))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [[str(tok).strip(), round(float(val), 4)] for tok, val in pairs[:n] if str(tok).strip()]


# ============================================================
# Feature engineering - MUST match each training script exactly
# ============================================================

def email_features(subject, body):
    text = f"{subject} {body}".strip()
    has_url = int(bool(re.search(r"https?://|www\.", text, re.I)))
    text_length = len(text)
    exclaim_count = text.count("!")
    urgency_flag = int(bool(re.search(
        r"\b(urgent|verify|immediately|suspend|password|click here|act now|limited time|winner|congratulations)\b",
        text, re.I
    )))
    engineered = np.array([[has_url, text_length, exclaim_count, urgency_flag]])
    return text, engineered


SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "signin", "banking",
    "confirm", "webscr", "ebayisapi", "paypal", "password", "suspend",
]


def ensure_scheme(url):
    if not re.match(r"^[a-zA-Z]+://", url):
        return "http://" + url
    return url


def url_features(url):
    raw = str(url)
    try:
        parsed = urlparse(ensure_scheme(raw))
        host = parsed.netloc or ""
        path = parsed.path or ""
    except ValueError:
        host = ""
        path = raw

    stripped = re.sub(r"^[a-zA-Z]+://", "", raw)

    url_length = len(stripped)
    num_dots = stripped.count(".")
    num_hyphens = stripped.count("-")
    num_at = stripped.count("@")
    num_digits = sum(c.isdigit() for c in stripped)
    num_subdirs = path.count("/")
    num_params = stripped.count("=") + stripped.count("&")
    has_ip = int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}", host)))
    num_subdomains = max(host.count(".") - 1, 0) if host else 0
    is_shortened = int(bool(re.search(
        r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly", stripped, re.I
    )))
    suspicious_word_count = sum(1 for w in SUSPICIOUS_WORDS if w in stripped.lower())
    unique_char_ratio = len(set(stripped)) / max(len(stripped), 1)

    engineered = np.array([[
        url_length, num_dots, num_hyphens, num_at, num_digits,
        num_subdirs, num_params, has_ip, num_subdomains,
        is_shortened, suspicious_word_count, unique_char_ratio
    ]])
    return stripped, engineered


def top_shap_like_features(model, feature_names, x_row, n=5):
    """
    Lightweight explainability: for tree models, approximate per-prediction
    'top contributing features' using (feature value * global feature importance)
    as a proxy ranking. Used for the email and URL models (XGBoost/RandomForest
    over TF-IDF + engineered features). SMS has proper LIME/SHAP explanations
    instead, since it now runs on DistilBERT (see explain_sms above).
    """
    importances = model.feature_importances_
    x_dense = np.asarray(x_row.todense()).flatten() if hasattr(x_row, "todense") else np.asarray(x_row).flatten()
    contribution_proxy = importances * x_dense
    top_idx = np.argsort(np.abs(contribution_proxy))[::-1][:n]
    return [(feature_names[i], float(contribution_proxy[i])) for i in top_idx if x_dense[i] != 0]


# ============================================================
# Routes
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/email", methods=["POST"])
def predict_email():
    data = request.get_json(force=True)
    subject = data.get("subject", "")
    body = data.get("body", "")

    text, engineered = email_features(subject, body)
    tfidf_vec = email_vectorizer.transform([text])
    x = hstack([tfidf_vec, csr_matrix(engineered)]).tocsr()

    proba = email_model.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    label = "Spam/Phishing" if pred == 1 else "Legitimate"
    confidence = float(proba[pred])

    feature_names = list(email_vectorizer.get_feature_names_out()) + \
        ["has_url", "text_length", "exclaim_count", "urgency_flag"]
    top_features = top_shap_like_features(email_model, feature_names, x)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "top_features": top_features,
    })

@app.route("/predict/sms", methods=["POST"])
def predict_sms():
    data = request.get_json(force=True)
    text_input = data.get("text", "")

    proba = sms_predict_proba([text_input])[0]
    pred = int(np.argmax(proba))
    label = SMS_CLASS_NAMES[pred]
    confidence = float(proba[pred])

    # Always explain the class the model actually predicted (pred), not a
    # hardcoded class - keeps top_features consistent with label/confidence.
    lime_result = explain_sms_lime(text_input, pred)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "top_features": lime_result,
        "xai": {"lime": lime_result},
    })

@app.route("/explain/sms/shap", methods=["POST"])
def explain_sms_shap_route():
    """On-demand endpoint - call this only when the user asks for a deeper
    explanation (e.g. clicks 'Show SHAP explanation' in the UI), not on every
    scan. This keeps /predict/sms fast while still making SHAP available for
    the demo/report."""
    data = request.get_json(force=True)
    text_input = data.get("text", "")

    try:
        shap_result = explain_sms_shap(text_input)
        return jsonify({"shap": shap_result})
    except Exception as e:
        return jsonify({"shap": [], "error": str(e)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    data = request.get_json(force=True)
    url_input = data.get("url", "")

    stripped, engineered = url_features(url_input)
    tfidf_vec = url_vectorizer.transform([stripped])
    x = hstack([tfidf_vec, csr_matrix(engineered)]).tocsr()

    proba = url_model.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    label = "Phishing" if pred == 1 else "Legitimate"
    confidence = float(proba[pred])

    feature_names = list(url_vectorizer.get_feature_names_out()) + [
        "url_length", "num_dots", "num_hyphens", "num_at", "num_digits",
        "num_subdirs", "num_params", "has_ip", "num_subdomains",
        "is_shortened", "suspicious_word_count", "unique_char_ratio",
    ]
    top_features = top_shap_like_features(url_model, feature_names, x)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "top_features": top_features,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)