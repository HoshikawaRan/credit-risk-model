# Credit Risk Scoring Model

Predicting loan default risk from applicant and loan characteristics — a full pipeline from raw data to a credit-score-based approval cutoff.

**Live site:** https://hoshikawaran.github.io/credit-risk-model/

## Overview

Using a public Kaggle credit risk dataset (32,581 loan applications), this project builds and compares two models:

- **Logistic regression** — an interpretable baseline with reviewable coefficients and p-values
- **Random forest** — a higher-accuracy ensemble, explained with SHAP

| Model | Test AUC | Accuracy Ratio |
|---|---|---|
| Logistic regression | 0.869 | 0.739 |
| Random forest | 0.930 | 0.860 |

The random forest also reaches a KS statistic of 0.723, indicating strong separation between defaulting and non-defaulting borrowers.

**Key finding:** converting predicted default probability into a 0–100 credit score and raising the minimum approval score from 70 to 90 drops the approval rate from 80.5% to 55.9%, while the default rate among approved borrowers falls from 6.6% to 3.7%. The right cutoff is a risk-policy decision, not a model-performance one — the site walks through this trade-off in detail.

## What's in this repo

- **`index.html`** — the full write-up: EDA, methodology, model diagnostics, SHAP explainability, and the credit-policy cutoff analysis, in English and Japanese.
- **`credit_risk_model_cleaned.ipynb`** — the full analysis notebook. Cleaned up from the original working notebook (duplicate cells, debug checks, and broken/superseded attempts removed), with every real output preserved exactly as originally produced.
- **`credit_risk_model_simple.py`** — the same pipeline rewritten as a single, linear, heavily-commented script, for anyone who wants a plainer walkthrough of the logic without jumping between notebook cells.

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle) — applicant demographics, loan terms, and credit history, with a binary default flag as the target.

## Built with

Python · pandas · scikit-learn · statsmodels · SHAP · matplotlib

## Author

Hoshikawa Ran
