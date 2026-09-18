"""
Credit Risk Scoring Model — simple, step-by-step version
==========================================================
Author: Hoshikawa Ran
Dataset: public Kaggle "Credit Risk Dataset" (credit_risk_dataset.csv)

This script is written to be READ, not just run. It follows one straight
path from raw data to a usable credit score, with a comment before each
step explaining *why* that step exists, not just what the code does.

The steps are:
  1. Load the data and check its quality
  2. Look at which factors are linked to default (exploratory analysis)
  3. Split the data into a training set and a test set
  4. Train a simple, interpretable model: logistic regression
  5. Train a more powerful model: random forest
  6. Compare the two models on data they have never seen
  7. Explain the random forest's predictions with SHAP
  8. Turn predicted risk into a credit score and a lending decision
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# A fixed random seed means anyone re-running this script gets the exact
# same train/test split and the exact same random forest.
RANDOM_STATE = 42


# ======================================================================
# STEP 1 — Load the data and check its quality
# ======================================================================
# Real-world data is never perfectly clean. Before building any model,
# it's worth looking at what's missing and what looks wrong.

df = pd.read_csv("credit_risk_dataset.csv")

print("Rows and columns:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# The oldest applicant on record is 144 years old — clearly a data entry
# error. In a real project you'd usually fix or remove rows like this.
# Here we leave it in on purpose, just to see how much (or little) it
# affects the model later on.
print("\nOldest applicant on record:", df["person_age"].max(), "(a data-entry error)")

# "loan_status" is the target: 1 means the borrower defaulted, 0 means
# they repaid. This is what we're trying to predict.
print("\nShare of borrowers who defaulted:", round(df["loan_status"].mean(), 3))


# ======================================================================
# STEP 2 — Which factors are linked to default?
# ======================================================================
# Before training any model, it helps to check the raw numbers: does
# default look more common for some groups than others? This also
# doubles as a sanity check — if nothing here matches intuition,
# something is probably wrong with the data.

print("\nDefault rate by loan grade (A = best, G = worst):")
print(df.groupby("loan_grade")["loan_status"].mean().round(3))

print("\nDefault rate by home ownership:")
print(df.groupby("person_home_ownership")["loan_status"].mean().round(3))

# "loan_percent_income" is the loan amount divided by the borrower's
# income — essentially, how big a bite the loan takes out of their
# earnings. We'll see later that this turns out to be the single
# strongest predictor of default in the whole dataset.
income_share_bins = [0, 0.10, 0.20, 0.30, 0.40, 1.0]
df["income_share_band"] = pd.cut(df["loan_percent_income"], bins=income_share_bins)
print("\nDefault rate by loan-to-income band:")
print(df.groupby("income_share_band", observed=True)["loan_status"].mean().round(3))


# ======================================================================
# STEP 3 — Split into training data and test data
# ======================================================================
# We train the model on most of the data (80%), then check how well it
# performs on the remaining 20% — data it has NEVER seen. This is the
# only honest way to know whether a model actually generalizes, rather
# than just memorizing the examples it was trained on.

numeric_columns = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
categorical_columns = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]

X = df[numeric_columns + categorical_columns]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,  # keep the same default rate in both the train and test sets
)

print(f"\nTraining on {len(X_train)} borrowers, testing on {len(X_test)}")


# ======================================================================
# STEP 4 — Train a simple, interpretable model: logistic regression
# ======================================================================
# Logistic regression is the traditional, industry-standard approach to
# credit scoring. It's not the most powerful model available, but every
# coefficient it produces can be explained to a regulator or a credit
# committee in one sentence — which matters a lot in this field.
#
# Before we can fit it, two preprocessing steps are needed:
#   - fill in missing numbers with the column median
#   - convert category columns (like "home ownership") into 0/1 columns,
#     since the model only understands numbers
numeric_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),  # puts all numeric columns on a similar scale
])
categorical_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encode", OneHotEncoder(handle_unknown="ignore")),
])
preprocessing = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

logistic_model = Pipeline([
    ("preprocessing", preprocessing),
    ("model", LogisticRegression(max_iter=1000)),
])
logistic_model.fit(X_train, y_train)

# predict_proba gives a probability between 0 and 1 for each borrower.
# We take column [1], the probability of the "default" class.
logistic_predicted_risk = logistic_model.predict_proba(X_test)[:, 1]


# ======================================================================
# STEP 5 — Train a more powerful model: random forest
# ======================================================================
# A random forest builds hundreds of decision trees on random slices of
# the data, then averages their votes. It can pick up on patterns that a
# straight-line model like logistic regression would miss — at the cost
# of being harder to explain (more on that in Step 7, with SHAP).
random_forest_preprocessing = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical", Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_columns),
])

random_forest_model = Pipeline([
    ("preprocessing", random_forest_preprocessing),
    ("model", RandomForestClassifier(
        n_estimators=500,       # number of trees
        min_samples_leaf=5,     # keeps individual trees from overfitting to single rows
        random_state=RANDOM_STATE,
        n_jobs=-1,               # use all available CPU cores
    )),
])
random_forest_model.fit(X_train, y_train)
forest_predicted_risk = random_forest_model.predict_proba(X_test)[:, 1]


# ======================================================================
# STEP 6 — Compare the two models on the test set
# ======================================================================
# AUC (Area Under the ROC Curve) answers a simple question: if you pick
# one defaulter and one non-defaulter at random, how often does the
# model correctly rank the defaulter as riskier? 0.5 = random guessing,
# 1.0 = perfect separation.

logistic_auc = roc_auc_score(y_test, logistic_predicted_risk)
forest_auc = roc_auc_score(y_test, forest_predicted_risk)

print(f"\nLogistic regression AUC on test data: {logistic_auc:.3f}")
print(f"Random forest AUC on test data:       {forest_auc:.3f}")

# A confusion matrix shows, at a 50% cutoff, how many borrowers were
# correctly vs. incorrectly classified.
forest_predicted_class = (forest_predicted_risk >= 0.50).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, forest_predicted_class).ravel()
print(f"\nRandom forest confusion matrix (test set, 0.5 cutoff):")
print(f"  Correctly approved (true negative):  {tn}")
print(f"  Wrongly declined (false positive):   {fp}")
print(f"  Wrongly approved (false negative):   {fn}")
print(f"  Correctly declined (true positive):  {tp}")

# A simple ROC plot: the closer the curve hugs the top-left corner,
# the better the model is at separating defaulters from safe borrowers.
plt.figure(figsize=(6, 5))
for label, risk in [("Logistic Regression", logistic_predicted_risk),
                     ("Random Forest", forest_predicted_risk)]:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, risk)
    plt.plot(false_positive_rate, true_positive_rate, label=label)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Test Set")
plt.legend()
plt.tight_layout()
plt.show()


# ======================================================================
# STEP 7 — Explain the random forest's predictions with SHAP
# ======================================================================
# The random forest is more accurate, but it's a "black box" — it
# doesn't hand you a simple formula. SHAP values fix that: for each
# borrower, they show how much each feature pushed their predicted risk
# up or down. Averaged across many borrowers, this also gives an
# importance ranking, similar in spirit to the random forest's built-in
# feature_importances_, but with the added benefit of showing direction.

# Running SHAP on the full test set can be slow, so we use a sample.
sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
sample_transformed = random_forest_model.named_steps["preprocessing"].transform(sample)
feature_names = random_forest_model.named_steps["preprocessing"].get_feature_names_out()
sample_df = pd.DataFrame(sample_transformed, columns=feature_names)

explainer = shap.TreeExplainer(random_forest_model.named_steps["model"])
shap_values = explainer.shap_values(sample_df)
# Different shap/sklearn versions return this in slightly different shapes;
# this line picks out the values for the "default" class either way.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

shap.summary_plot(shap_values, sample_df, max_display=10)


# ======================================================================
# STEP 8 — Turn predicted risk into a credit score and a decision
# ======================================================================
# A predicted probability like "0.07" isn't very intuitive to act on.
# Flipping it into a 0–100 score, where 100 is the safest possible
# applicant, makes it easy to set a plain-language approval rule:
# "approve anyone scoring above X."

credit_score = 100 * (1 - forest_predicted_risk)

print("\nWhat happens at different approval cutoffs:")
for cutoff in [50, 70, 90]:
    approved = credit_score >= cutoff
    approval_rate = approved.mean()
    default_rate_if_approved = y_test[approved].mean()
    print(
        f"  Cutoff {cutoff:>3}: approve {approval_rate:.1%} of applicants, "
        f"default rate among them = {default_rate_if_approved:.1%}"
    )

# The trade-off is unavoidable: a stricter cutoff approves fewer people,
# but the people it does approve are, on average, safer. Where exactly
# to set that cutoff is a business decision, not something the model can
# answer on its own — it depends on how much risk the lender is willing
# to accept in exchange for how much lending volume.
