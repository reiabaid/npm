"""
SCOPE – Model Training
======================
Pipeline:
  1. Load dataset → preprocess (ColumnTransformer) → SMOTE balance
  2. 5-fold stratified CV to compare RandomForest and XGBoost
  3. GridSearchCV on the better model to tune hyperparameters
  4. Threshold tuning: pick the cut-off that maximises recall ≥ 0.90
  5. Final evaluation on the held-out test set
  6. Save: scope_model.joblib, scope_preprocessor.joblib, scope_threshold.json

Run from the project root:
    python -m src.model.train
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost candidate.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.model.preprocess import build_preprocessor  # noqa: E402

DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

NUMERICAL_FEATURES = [
    "days_since_created", "days_since_last_update", "num_versions",
    "release_velocity", "num_maintainers", "description_length",
    "weekly_downloads", "typosquat_min_distance", "script_suspicion_score",
    "maintainer_min_account_age_days",
    "stargazers_count", "forks_count", "open_issues_count",
    "subscribers_count", "contributor_count", "days_since_last_commit",
]
BINARY_FEATURES    = ["has_any_install_hook", "license_is_standard", "has_github_repo"]
ALL_FEATURE_NAMES  = NUMERICAL_FEATURES + BINARY_FEATURES

# ── Target recall floor for threshold selection (security bias) ──────
MIN_RECALL = 0.90


# ─────────────────────────────────────────────────────────────────────
# 1. Load & split
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SCOPE – Model Training")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    sys.exit(f"[ERROR] Dataset not found at {DATASET_PATH}.\n"
             "Run src/data/build_dataset.py first.")

print(f"\n[1] Dataset  shape={df.shape}")

X = df[ALL_FEATURE_NAMES]
y = df["label"]

# 15 % stratified hold-out — never used for model selection
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
print(f"    dev={len(X_dev)}  test={len(X_test)}")


# ─────────────────────────────────────────────────────────────────────
# 2. Preprocess (fit on dev only — no leakage into test)
# ─────────────────────────────────────────────────────────────────────
print("\n[2] Fitting preprocessor on dev set …")
preprocessor = build_preprocessor()
X_dev_t  = preprocessor.fit_transform(X_dev)
X_test_t = preprocessor.transform(X_test)

joblib.dump(preprocessor, os.path.join(MODELS_DIR, "scope_preprocessor.joblib"))
print("    Saved → models/scope_preprocessor.joblib")


# ─────────────────────────────────────────────────────────────────────
# 3. SMOTE on dev set
# ─────────────────────────────────────────────────────────────────────
print("\n[3] Applying SMOTE …")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_dev_t, y_dev)
counts = pd.Series(y_res).value_counts().sort_index()
print(f"    After SMOTE → class 0: {counts[0]}  class 1: {counts[1]}")


# ─────────────────────────────────────────────────────────────────────
# 4. 5-fold CV to pick the best model family
# ─────────────────────────────────────────────────────────────────────
print("\n[4] 5-fold CV comparison (scoring = f1 on suspicious class) …")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}
if _HAS_XGB:
    candidates["XGBoost"] = XGBClassifier(
        n_estimators=200, random_state=42,
        eval_metric="logloss", verbosity=0,
    )

cv_scores = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring="f1", n_jobs=-1)
    cv_scores[name] = scores
    print(f"    {name:<20} CV F1 = {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
print(f"\n    Winner: {best_name}")


# ─────────────────────────────────────────────────────────────────────
# 5. GridSearchCV on the winning family
# ─────────────────────────────────────────────────────────────────────
print(f"\n[5] GridSearchCV on {best_name} …")

if best_name == "RandomForest":
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth":    [None, 15, 30],
        "min_samples_split": [2, 5],
    }
    base_clf = RandomForestClassifier(random_state=42)
else:
    param_grid = {
        "n_estimators":  [200, 400],
        "max_depth":     [4, 8],
        "learning_rate": [0.05, 0.1],
    }
    base_clf = XGBClassifier(
        random_state=42, eval_metric="logloss", verbosity=0
    )

grid = GridSearchCV(
    base_clf, param_grid, cv=cv, scoring="f1",
    n_jobs=-1, verbose=1, refit=True,
)
grid.fit(X_res, y_res)

best_model = grid.best_estimator_
print(f"    Best params : {grid.best_params_}")
print(f"    Best CV F1  : {grid.best_score_:.4f}")


# ─────────────────────────────────────────────────────────────────────
# 6. Threshold tuning — maximise recall with recall ≥ MIN_RECALL
# ─────────────────────────────────────────────────────────────────────
print(f"\n[6] Threshold tuning (target recall ≥ {MIN_RECALL}) …")

y_dev_proba = best_model.predict_proba(X_dev_t)[:, 1]
precision_arr, recall_arr, thresholds = precision_recall_curve(y_dev, y_dev_proba)

# Among thresholds where recall ≥ MIN_RECALL, pick the one with highest F1
best_thresh = 0.5
best_f1_at_thresh = 0.0
for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds):
    if r >= MIN_RECALL:
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1_at_thresh:
            best_f1_at_thresh = f1
            best_thresh = float(t)

print(f"    Chosen threshold : {best_thresh:.4f}")
print(f"    F1 at threshold  : {best_f1_at_thresh:.4f}")

threshold_path = os.path.join(MODELS_DIR, "scope_threshold.json")
with open(threshold_path, "w") as fh:
    json.dump({"threshold": best_thresh}, fh)
print(f"    Saved → models/scope_threshold.json")


# ─────────────────────────────────────────────────────────────────────
# 7. Final test-set evaluation
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[7] FINAL TEST-SET EVALUATION")
print("=" * 60)

y_test_proba = best_model.predict_proba(X_test_t)[:, 1]
y_test_pred  = (y_test_proba >= best_thresh).astype(int)

print(classification_report(y_test, y_test_pred, target_names=["Healthy", "Suspicious"]))

cm = confusion_matrix(y_test, y_test_pred)
TN, FP, FN, TP = cm.ravel()
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
auc = roc_auc_score(y_test, y_test_proba)
f1  = f1_score(y_test, y_test_pred)

print(f"  Threshold used : {best_thresh:.4f}")
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  F1 (suspicious): {f1:.4f}")
print(f"  FNR            : {fnr*100:.1f}%  ({FN} missed)")
print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")

report_path = os.path.join(REPORTS_DIR, "final_evaluation.txt")
with open(report_path, "w") as fh:
    fh.write(f"Model        : {best_name}\n")
    fh.write(f"Best params  : {grid.best_params_}\n")
    fh.write(f"Threshold    : {best_thresh:.4f}\n")
    fh.write(f"ROC-AUC      : {auc:.4f}\n")
    fh.write(f"F1 (sus)     : {f1:.4f}\n")
    fh.write(f"FNR          : {fnr*100:.1f}%\n")
    fh.write(f"TN={TN}  FP={FP}  FN={FN}  TP={TP}\n\n")
    fh.write(classification_report(y_test, y_test_pred,
                                   target_names=["Healthy", "Suspicious"]))
print(f"\n    Saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────────────────────────────────
print("\n[8] Generating plots …")

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Suspicious"]).plot(ax=ax_cm)
ax_cm.set_title(f"Confusion Matrix — {best_name} (Test Set, threshold={best_thresh:.2f})")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_final.png"), dpi=150)
print("    Saved → reports/figures/confusion_matrix_final.png")

fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=ax_roc, name=best_name)
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax_roc.set_title(f"ROC Curve — {best_name} (Test Set, AUC={auc:.4f})")
fig_roc.tight_layout()
fig_roc.savefig(os.path.join(PLOTS_DIR, "roc_curve_final.png"), dpi=150)
print("    Saved → reports/figures/roc_curve_final.png")


# ─────────────────────────────────────────────────────────────────────
# 9. Save model
# ─────────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "scope_model.joblib")
joblib.dump(best_model, model_path)
print(f"\n[9] Model saved → models/scope_model.joblib")

plt.show()
print("\nDone.")
