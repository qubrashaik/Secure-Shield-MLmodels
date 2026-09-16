# SecureShield — URL Phishing Detection Model

## Files
- `url_data_prep.py` — merges `phishing_site_urls.csv` (good/bad) and
  `PhishTank_2026.csv` (confirmed phishing), dedupes, engineers lexical/
  structural URL features.
- `train_url_model.py` — trains character n-gram TF-IDF (analyzer="char_wb",
  3-5 grams) + engineered features -> RandomForest, evaluates, saves artifacts.
- `url_model_artifacts/`
  - `char_tfidf_vectorizer.pkl`, `url_model.pkl`
  - `feature_importances.csv`, `metrics.json`

## Important data quality fix (worth mentioning in your report)
The two source datasets had a hidden confound: every URL in `PhishTank_2026.csv`
(100% phishing) includes an `https://` scheme, while `phishing_site_urls.csv`
(mixed good/bad) never includes any scheme at all. Naively combining them made
"has `https://`" look like a strong phishing predictor — it was actually just
identifying which file a row came from, not a real phishing signal (a form of
data leakage). The scheme is now stripped before feature extraction and the
`has_https` feature was dropped entirely, since it can't be estimated
reliably from this data. The reported metrics below are from the corrected model.

## Results (120,000 URLs, balanced 60K/class downsample, 80/20 split)
- Accuracy: 89.8%
- Precision: 94.8%
- Recall: 84.3%
- F1: 89.2%
- ROC-AUC: 0.97

Top signals: suspicious keyword count (login/verify/secure/etc.), `.com/`-type
character patterns, digit count, dot/hyphen count, URL length, subdomain count
— all genuine structural/lexical phishing indicators.

## Scaling up on your own machine
This sandbox (1 CPU / ~4GB RAM) required downsampling to 120K of the full
571K deduped URLs. On your machine, in `url_data_prep.py`, call:
```python
df = build_dataset(sample_per_class=None)  # use all ~393K legit / 179K phishing
```
and increase `max_features` in the TF-IDF vectorizer (e.g. 8000-10000) and
`n_estimators` in RandomForest (e.g. 300-500) for better accuracy.

## Upgrading to XGBoost + SHAP (same as email/SMS models)
Same swap-in steps as the other two models. Load `url_model.pkl` +
`char_tfidf_vectorizer.pkl` in your prediction microservice at a
`/predict/url` endpoint.

## All three models now follow the same architecture
Python microservice (Flask/FastAPI) with three endpoints —
`/predict/email`, `/predict/sms`, `/predict/url` — each loading its own
vectorizer + model, returning `{label, confidence, top_features}`.
Spring Boot calls these via REST, stores results in MySQL, and your
React dashboard renders scan history + stats from the Spring Boot APIs.
