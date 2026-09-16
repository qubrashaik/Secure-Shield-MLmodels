import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lime.lime_text import LimeTextExplainer
import shap

from urllib.parse import urlparse


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL DIRECTORIES
# ============================================================

EMAIL_DIR = os.path.join(
    BASE_DIR,
    "emaildev",
    "model_artifacts"
)

SMS_DIR = os.path.join(
    BASE_DIR,
    "smsdev",
    "model_artifacts"
)

SMS_DISTILBERT_DIR = os.path.join(
    BASE_DIR,
    "smsdev",
    "model_artifacts_distilbert"
)

URL_DIR = os.path.join(
    BASE_DIR,
    "urldev",
    "model_artifacts"
)


# ============================================================
# CLASS NAMES
# ============================================================

EMAIL_CLASS_NAMES = [
    "Legitimate",
    "Spam",
    "Phishing"
]

SMS_CLASS_NAMES = [
    "Legitimate",
    "Spam",
    "Phishing"
]

URL_CLASS_NAMES = [
    "Legitimate",
    "Spam",
    "Phishing"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def check_file(path):
    """
    Check whether a model file/folder exists.
    """
    if not os.path.exists(path):
        print(f"WARNING: Path not found -> {path}")
        return False

    return True


# ============================================================
# LOAD EMAIL MODEL
# ============================================================

print("==========================================")
print("Loading Email model...")
print("==========================================")

EMAIL_VECTORIZER_PATH = os.path.join(
    EMAIL_DIR,
    "tfidf_vectorizer.pkl"
)

EMAIL_MODEL_PATH = os.path.join(
    EMAIL_DIR,
    "email_model.pkl"
)

check_file(EMAIL_VECTORIZER_PATH)
check_file(EMAIL_MODEL_PATH)

email_vectorizer = joblib.load(
    EMAIL_VECTORIZER_PATH
)

email_model = joblib.load(
    EMAIL_MODEL_PATH
)

print("Email model loaded successfully.")


# ============================================================
# LOAD SMS DISTILBERT MODEL
# ============================================================

print("==========================================")
print("Loading SMS DistilBERT model...")
print("==========================================")

check_file(SMS_DISTILBERT_DIR)

sms_tokenizer = AutoTokenizer.from_pretrained(
    SMS_DISTILBERT_DIR
)

sms_model = AutoModelForSequenceClassification.from_pretrained(
    SMS_DISTILBERT_DIR
)

sms_model.eval()

print("SMS DistilBERT model loaded successfully.")


# ============================================================
# LOAD URL MODEL
# ============================================================

print("==========================================")
print("Loading URL model...")
print("==========================================")

URL_VECTORIZER_PATH = os.path.join(
    URL_DIR,
    "char_tfidf_vectorizer.pkl"
)

URL_MODEL_PATH = os.path.join(
    URL_DIR,
    "url_model.pkl"
)

check_file(URL_VECTORIZER_PATH)
check_file(URL_MODEL_PATH)

url_vectorizer = joblib.load(
    URL_VECTORIZER_PATH
)

url_model = joblib.load(
    URL_MODEL_PATH
)

print("URL model loaded successfully.")


print("==========================================")
print("ALL MODELS LOADED SUCCESSFULLY")
print("==========================================")


# ============================================================
# LIME EXPLAINER FOR SMS
# ============================================================

lime_sms_explainer = LimeTextExplainer(
    class_names=SMS_CLASS_NAMES
)


# ============================================================
# SMS PREDICTION FUNCTION
# Used by LIME and SHAP
# ============================================================

def sms_predict_proba(texts):

    if isinstance(texts, str):
        texts = [texts]

    inputs = sms_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = sms_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    return probabilities.cpu().numpy()


# ============================================================
# SHAP EXPLAINER
# ============================================================

try:

    shap_sms_masker = shap.maskers.Text(
        sms_tokenizer
    )

    shap_sms_explainer = shap.Explainer(
        sms_predict_proba,
        shap_sms_masker
    )

    print("SMS SHAP explainer initialized.")

except Exception as e:

    print("SHAP initialization warning:")
    print(e)

    shap_sms_explainer = None


# ============================================================
# EMAIL FEATURE ENGINEERING
# ============================================================

def create_email_features(subject, body):

    subject = str(subject or "")
    body = str(body or "")

    text = subject + " " + body

    features = {
        "text_length": len(text),

        "subject_length": len(subject),

        "body_length": len(body),

        "word_count": len(
            text.split()
        ),

        "digit_count": sum(
            char.isdigit()
            for char in text
        ),

        "uppercase_count": sum(
            char.isupper()
            for char in text
        ),

        "exclamation_count": text.count("!"),

        "question_count": text.count("?"),

        "has_url": int(
            bool(
                re.search(
                    r"https?://|www\.",
                    text,
                    re.IGNORECASE
                )
            )
        ),

        "has_email": int(
            bool(
                re.search(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                    text
                )
            )
        ),

        "has_phone_number": int(
            bool(
                re.search(
                    r"\b\d{10}\b",
                    text
                )
            )
        ),

        "has_currency": int(
            bool(
                re.search(
                    r"[$₹€£]",
                    text
                )
            )
        ),

        "has_html": int(
            bool(
                re.search(
                    r"<[^>]+>",
                    text
                )
            )
        )
    }

    return pd.DataFrame([features])


# ============================================================
# URL FEATURE ENGINEERING
# ============================================================

def create_url_features(url):

    url = str(url or "")

    parsed = urlparse(
        url if "://" in url else "http://" + url
    )

    hostname = parsed.netloc

    if not hostname:
        hostname = parsed.path.split("/")[0]

    path = parsed.path

    features = {

        "url_length": len(url),

        "hostname_length": len(hostname),

        "path_length": len(path),

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_underscores": url.count("_"),

        "num_slashes": url.count("/"),

        "num_question_marks": url.count("?"),

        "num_equals": url.count("="),

        "num_ampersands": url.count("&"),

        "num_digits": sum(
            char.isdigit()
            for char in url
        ),

        "has_https": int(
            url.lower().startswith("https")
        ),

        "has_ip": int(
            bool(
                re.search(
                    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
                    url,
                    re.IGNORECASE
                )
            )
        ),

        "has_at_symbol": int(
            "@" in url
        ),

        "has_percent": int(
            "%" in url
        ),

        "has_double_slash": int(
            "//" in url.replace("://", "")
        )
    }

    return pd.DataFrame([features])


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

def get_model_feature_importance(
    model,
    feature_names,
    feature_values,
    top_n=10
):

    try:

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            return []

        contribution = (
            np.asarray(importances)
            *
            np.abs(
                np.asarray(feature_values)
            )
        )

        indices = np.argsort(
            contribution
        )[::-1][:top_n]

        result = []

        for index in indices:

            result.append({
                "feature": str(
                    feature_names[index]
                ),

                "importance": float(
                    importances[index]
                ),

                "contribution": float(
                    contribution[index]
                )
            })

        return result

    except Exception as e:

        print(
            "Feature importance error:",
            e
        )

        return []


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "SecureShield ML API",
        "status": "running",
        "models": {
            "email": "XGBoost",
            "sms": "DistilBERT",
            "url": "XGBoost"
        }
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "email_model": "loaded",
        "sms_model": "loaded",
        "url_model": "loaded"
    })


# ============================================================
# EMAIL PREDICTION
# ============================================================

@app.route(
    "/predict/email",
    methods=["POST"]
)
def predict_email():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is empty."
            }), 400

        subject = str(
            data.get("subject", "")
        )

        body = str(
            data.get("body", "")
        )

        if not subject and not body:

            return jsonify({
                "error": "Subject or body is required."
            }), 400

        text = (
            subject
            + " "
            + body
        )

        # ----------------------------------------
        # TF-IDF
        # ----------------------------------------

        text_vector = email_vectorizer.transform(
            [text]
        )

        # ----------------------------------------
        # Engineered features
        # ----------------------------------------

        engineered_df = create_email_features(
            subject,
            body
        )

        # ----------------------------------------
        # Combine features
        # ----------------------------------------

        engineered_values = sp.csr_matrix(
            engineered_df.values
        )

        final_features = sp.hstack([
            text_vector,
            engineered_values
        ])

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        prediction = email_model.predict(
            final_features
        )[0]

        probabilities = email_model.predict_proba(
            final_features
        )[0]

        predicted_class = int(
            prediction
        )

        confidence = float(
            np.max(probabilities)
        )

        # ----------------------------------------
        # Label mapping
        # ----------------------------------------

        if predicted_class < len(
            EMAIL_CLASS_NAMES
        ):

            label = EMAIL_CLASS_NAMES[
                predicted_class
            ]

        else:

            label = str(
                predicted_class
            )

        # ----------------------------------------
        # Feature importance
        # ----------------------------------------

        feature_names = list(
            email_vectorizer.get_feature_names_out()
        )

        feature_names += list(
            engineered_df.columns
        )

        feature_values = np.asarray(
            final_features.toarray()[0]
        )

        top_features = (
            get_model_feature_importance(
                email_model,
                feature_names,
                feature_values,
                top_n=10
            )
        )

        return jsonify({

            "type": "email",

            "label": label,

            "confidence": round(
                confidence,
                4
            ),

            "probabilities": {
                EMAIL_CLASS_NAMES[i]:
                    round(
                        float(probabilities[i]),
                        4
                    )

                for i in range(
                    min(
                        len(probabilities),
                        len(EMAIL_CLASS_NAMES)
                    )
                )
            },

            "top_features": top_features

        })

    except Exception as e:

        print(
            "Email prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# SMS PREDICTION
# ============================================================

@app.route(
    "/predict/sms",
    methods=["POST"]
)
def predict_sms():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is empty."
            }), 400

        text = str(
            data.get("text", "")
        )

        if not text.strip():

            return jsonify({
                "error": "SMS text is required."
            }), 400

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        probabilities = sms_predict_proba(
            [text]
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        # ----------------------------------------
        # Label
        # ----------------------------------------

        if predicted_class < len(
            SMS_CLASS_NAMES
        ):

            label = SMS_CLASS_NAMES[
                predicted_class
            ]

        else:

            label = str(
                predicted_class
            )

        # ----------------------------------------
        # LIME explanation
        # ----------------------------------------

        lime_explanation = []

        try:

            explanation = lime_sms_explainer.explain_instance(
                text,
                sms_predict_proba,
                num_features=10
            )

            for word, weight in (
                explanation.as_list()
            ):

                # Make positive values mean
                # movement toward suspicious class
                adjusted_weight = float(
                    weight
                )

                if predicted_class == 0:
                    adjusted_weight *= -1

                lime_explanation.append({

                    "feature": word,

                    "weight": round(
                        adjusted_weight,
                        5
                    )
                })

        except Exception as lime_error:

            print(
                "LIME error:",
                lime_error
            )

        return jsonify({

            "type": "sms",

            "label": label,

            "confidence": round(
                confidence,
                4
            ),

            "probabilities": {
                SMS_CLASS_NAMES[i]:
                    round(
                        float(probabilities[i]),
                        4
                    )

                for i in range(
                    min(
                        len(probabilities),
                        len(SMS_CLASS_NAMES)
                    )
                )
            },

            "xai": {

                "method": "LIME",

                "features": lime_explanation

            }

        })

    except Exception as e:

        print(
            "SMS prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# SMS SHAP EXPLANATION
# ============================================================

@app.route(
    "/explain/sms/shap",
    methods=["POST"]
)
def explain_sms_shap():

    try:

        if shap_sms_explainer is None:

            return jsonify({
                "error": "SHAP explainer is not available."
            }), 500

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is empty."
            }), 400

        text = str(
            data.get("text", "")
        )

        if not text.strip():

            return jsonify({
                "error": "SMS text is required."
            }), 400

        # ----------------------------------------
        # Generate SHAP values
        # ----------------------------------------

        shap_values = shap_sms_explainer(
            [text]
        )

        values = shap_values.values[0]

        data_values = shap_values.data[0]

        # ----------------------------------------
        # Handle output shape
        # ----------------------------------------

        if len(values.shape) == 2:

            predicted_probabilities = (
                sms_predict_proba([text])[0]
            )

            predicted_class = int(
                np.argmax(
                    predicted_probabilities
                )
            )

            token_values = values[
                :,
                predicted_class
            ]

        else:

            token_values = values

        # ----------------------------------------
        # Build explanation
        # ----------------------------------------

        shap_features = []

        for token, value in zip(
            data_values,
            token_values
        ):

            if token is None:
                continue

            token = str(token).strip()

            if not token:
                continue

            shap_features.append({

                "feature": token,

                "shap_value": round(
                    float(value),
                    6
                )

            })

        # ----------------------------------------
        # Sort by absolute importance
        # ----------------------------------------

        shap_features = sorted(
            shap_features,
            key=lambda x:
                abs(x["shap_value"]),
            reverse=True
        )[:15]

        return jsonify({

            "type": "sms",

            "method": "SHAP",

            "features": shap_features

        })

    except Exception as e:

        print(
            "SMS SHAP error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# URL PREDICTION
# ============================================================

@app.route(
    "/predict/url",
    methods=["POST"]
)
def predict_url():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is empty."
            }), 400

        url = str(
            data.get("url", "")
        )

        if not url.strip():

            return jsonify({
                "error": "URL is required."
            }), 400

        # ----------------------------------------
        # Character TF-IDF
        # ----------------------------------------

        text_vector = url_vectorizer.transform(
            [url]
        )

        # ----------------------------------------
        # Engineered URL features
        # ----------------------------------------

        engineered_df = create_url_features(
            url
        )

        engineered_values = sp.csr_matrix(
            engineered_df.values
        )

        # ----------------------------------------
        # Combine
        # ----------------------------------------

        final_features = sp.hstack([
            text_vector,
            engineered_values
        ])

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        prediction = url_model.predict(
            final_features
        )[0]

        probabilities = url_model.predict_proba(
            final_features
        )[0]

        predicted_class = int(
            prediction
        )

        confidence = float(
            np.max(probabilities)
        )

        # ----------------------------------------
        # Label
        # ----------------------------------------

        if predicted_class < len(
            URL_CLASS_NAMES
        ):

            label = URL_CLASS_NAMES[
                predicted_class
            ]

        else:

            label = str(
                predicted_class
            )

        # ----------------------------------------
        # Feature importance
        # ----------------------------------------

        feature_names = list(
            url_vectorizer.get_feature_names_out()
        )

        feature_names += list(
            engineered_df.columns
        )

        feature_values = np.asarray(
            final_features.toarray()[0]
        )

        top_features = (
            get_model_feature_importance(
                url_model,
                feature_names,
                feature_values,
                top_n=10
            )
        )

        return jsonify({

            "type": "url",

            "url": url,

            "label": label,

            "confidence": round(
                confidence,
                4
            ),

            "probabilities": {
                URL_CLASS_NAMES[i]:
                    round(
                        float(probabilities[i]),
                        4
                    )

                for i in range(
                    min(
                        len(probabilities),
                        len(URL_CLASS_NAMES)
                    )
                )
            },

            "top_features": top_features

        })

    except Exception as e:

        print(
            "URL prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Endpoint not found."
    }), 404


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )