# SecureShield — Email Phishing/Spam Model

## Files
- `data_prep.py` — loads your 7 raw CSVs (Assassin, CEAS-08, Enron, Ling, TREC-05/06/07),
  standardizes columns, cleans malformed label rows, merges into one dataset, and
  engineers basic features (`has_url`, `text_length`, `exclaim_count`, `urgency_flag`).
- `train_email_model.py` — trains the classifier (TF-IDF + engineered features → RandomForest),
  evaluates it, and saves the model artifacts.
- `model_artifacts/`
  - `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
  - `email_model.pkl` — trained classifier
  - `feature_importances.csv` — ranked feature importances
  - `metrics.json` — accuracy/precision/recall/F1/ROC-AUC + confusion matrix

## How to run
1. Put your 7 CSV files in a folder, update `DATA_DIR` in `data_prep.py`.
2. `python data_prep.py` → produces `merged_email_dataset.csv`.
3. `python train_email_model.py` → trains and saves the model.

## Results on this run (202,917 emails, 80/20 split)
- Accuracy: 94.7%
- Precision: 91.7%
- Recall: 97.5%  (important — missing phishing is worse than a false alarm)
- F1: 94.5%
- ROC-AUC: 0.99

## Upgrading to match your full tech stack (XGBoost + SHAP)
This sandbox has no internet access, so it couldn't install `xgboost`/`shap`.
RandomForest was used instead (same idea, similar performance, built-in
feature importances). On your own machine:

```bash
pip install xgboost shap
```

Then in `train_email_model.py`, replace the RandomForest block with:

```python
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    eval_metric="logloss", random_state=42
)
model.fit(X_train_full, y_train)
```

For SHAP explainability (for your XAI requirement / dashboard):

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_full)

# For a single prediction, get top contributing features:
def explain_prediction(idx):
    contributions = shap_values[idx]
    top_idx = np.argsort(np.abs(contributions))[::-1][:5]
    return [(feature_names[i], contributions[i]) for i in top_idx]
```

This gives you, per email: **label, confidence score (from `predict_proba`),
and top contributing features** — exactly the format your Spring Boot
backend needs to feed the dashboard.

## Also on your local machine (more RAM/CPU available)
This sandbox only had 1 CPU / ~4GB RAM, so settings were kept conservative
(`max_features=3000`, `n_estimators=100`, `max_depth=25`). On a normal
machine you can push these higher for better accuracy:
- TF-IDF `max_features=8000-10000`, `ngram_range=(1,2)`
- RandomForest/XGBoost `n_estimators=300-500`, `max_depth=None`

## Next steps
- Same pipeline structure (TF-IDF + engineered features + tree model + SHAP)
  can be reused for the SMS and URL models — just swap the feature engineering
  step for channel-specific signals.
