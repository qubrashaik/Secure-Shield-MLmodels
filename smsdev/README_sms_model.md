# SecureShield — SMS Spam/Phishing Model

## Files
- `sms_data_prep.py` — loads `SMSSpamCollection`, cleans it, engineers
  SMS-specific features (digit_count, has_phone_number, has_currency,
  uppercase_ratio, urgency_flag, has_url, exclaim_count).
- `train_sms_model.py` — trains TF-IDF + RandomForest classifier, evaluates,
  saves artifacts.
- `sms_model_artifacts/`
  - `tfidf_vectorizer.pkl`, `sms_model.pkl`
  - `feature_importances.csv`, `metrics.json`

## Results (5,572 SMS messages, 80/20 split)
- Accuracy: 98.2%
- Precision: 93.9%
- Recall: 92.6%
- F1: 93.2%
- ROC-AUC: 0.99

Digit count, presence of a phone number, currency/prize words, and
uppercase ratio were the strongest signals — matches how SMS spam
typically works (short urgent messages pushing you to call/text a number).

## Upgrading to XGBoost + SHAP (same as email model)
See the email model's README — identical swap-in steps apply here.
Load `sms_model.pkl` + `tfidf_vectorizer.pkl` the same way in your
prediction microservice, at a `/predict/sms` endpoint.

## Same overall architecture as the email model
Python microservice (Flask/FastAPI) loads this model → exposes
`/predict/sms` → Spring Boot calls it via REST → stores result in
MySQL → React dashboard displays it.
