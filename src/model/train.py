"""
SCOPE â€“ Model Training
======================
Pipeline:
  1. Load dataset â†’ preprocess (ColumnTransformer) â†’ SMOTE balance
  2. 5-fold stratified CV to compare RandomForest and XGBoost
  3. GridSearchCV on the better model to tune hyperparameters
  4. Threshold tuning: pick the cut-off that maximises recall â‰¥ 0.90
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
from sklearn.isotonic import IsotonicRegression
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
    print("[WARN] xgboost not installed â€” skipping XGBoost candidate.")

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

# â”€â”€ Target recall floor for threshold selection (security bias) â”€â”€â”€â”€â”€â”€
MIN_RECALL = 0.90


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Load & split
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("SCOPE â€“ Model Training")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    sys.exit(f"[ERROR] Dataset not found at {DATASET_PATH}.\n"
             "Run src/data/build_dataset.py first.")

print(f"\n[1] Dataset  shape={df.shape}")

X = df[ALL_FEATURE_NAMES]
y = df["label"]

# 15 % stratified hold-out â€” never used for model selection
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
print(f"    dev={len(X_dev)}  test={len(X_test)}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Preprocess (fit on dev only â€” no leakage into test)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[2] Fitting preprocessor on dev set â€¦")
preprocessor = build_preprocessor()
X_dev_t  = preprocessor.fit_transform(X_dev)
X_test_t = preprocessor.transform(X_test)

joblib.dump(preprocessor, os.path.join(MODELS_DIR, "scope_preprocessor.joblib"))
print("    Saved â†’ models/scope_preprocessor.joblib")


# ─────────────────────────────────────────────────────────────────────
# 3. Split dev into model-train (80%) / cal (20%), then SMOTE train
#    Cal split is held out from SMOTE so calibrator sees out-of-sample
#    predictions — prevents isotonic regression from memorising train proba.
# ─────────────────────────────────────────────────────────────────────
print("\n[3] Splitting dev -> train (80%) / cal (20%), then SMOTE...")
X_train_t, X_cal_t, y_train, y_cal = train_test_split(
    X_dev_t, y_dev, test_size=0.20, stratify=y_dev, random_state=42
)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_t, y_train)
counts = pd.Series(y_res).value_counts().sort_index()
print(f"    Train {len(X_train_t)} / Cal {len(X_cal_t)} | After SMOTE -> class 0: {counts[0]}  class 1: {counts[1]}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. 5-fold CV to pick the best model family
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[4] 5-fold CV comparison (scoring = f1 on suspicious class) â€¦")

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
    print(f"    {name:<20} CV F1 = {scores.mean():.4f} Â± {scores.std():.4f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
print(f"\n    Winner: {best_name}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. GridSearchCV on the winning family
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[5] GridSearchCV on {best_name} â€¦")

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5b. Probability calibration (separate isotonic calibrator)
#     Keeps the original tree model intact for SHAP compatibility.
#     The calibrator maps raw predict_proba output â†’ calibrated score.
#     Calibrate on original dev set (not SMOTE'd) â€” real class distribution.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[5b] Fitting isotonic calibrator on held-out cal set...")
raw_cal_proba = best_model.predict_proba(X_cal_t)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_cal_proba, y_cal)
calibrator_path = os.path.join(MODELS_DIR, "scope_calibrator.joblib")
joblib.dump(calibrator, calibrator_path)
print(f"     Saved â†’ models/scope_calibrator.joblib")
print(f"     Raw score range on cal: [{raw_cal_proba.min():.3f}, {raw_cal_proba.max():.3f}]")
cal_cal_proba = calibrator.transform(raw_cal_proba)
print(f"     Calibrated range on cal: [{cal_cal_proba.min():.3f}, {cal_cal_proba.max():.3f}]")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Threshold tuning â€” maximise recall with recall â‰¥ MIN_RECALL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[6] Threshold tuning (target recall â‰¥ {MIN_RECALL}) â€¦")

y_cal_proba = calibrator.transform(best_model.predict_proba(X_cal_t)[:, 1])
precision_arr, recall_arr, thresholds = precision_recall_curve(y_cal, y_cal_proba)

# Among thresholds where recall â‰¥ MIN_RECALL, pick the one with highest F1
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
print(f"    Saved â†’ models/scope_threshold.json")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. Final test-set evaluation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("[7] FINAL TEST-SET EVALUATION")
print("=" * 60)

y_test_proba = calibrator.transform(best_model.predict_proba(X_test_t)[:, 1])
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
    fh.write(f"Model        : {best_name} (calibrated with isotonic regression)\n")
    fh.write(f"Best params  : {grid.best_params_}\n")
    fh.write(f"Threshold    : {best_thresh:.4f}\n")
    fh.write(f"ROC-AUC      : {auc:.4f}\n")
    fh.write(f"F1 (sus)     : {f1:.4f}\n")
    fh.write(f"FNR          : {fnr*100:.1f}%\n")
    fh.write(f"TN={TN}  FP={FP}  FN={FN}  TP={TP}\n\n")
    fh.write(classification_report(y_test, y_test_pred,
                                   target_names=["Healthy", "Suspicious"]))
print(f"\n    Saved â†’ {report_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. Plots
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[8] Generating plots â€¦")

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Suspicious"]).plot(ax=ax_cm)
ax_cm.set_title(f"Confusion Matrix â€” {best_name} (Test Set, threshold={best_thresh:.2f})")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_final.png"), dpi=150)
print("    Saved â†’ reports/figures/confusion_matrix_final.png")

fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=ax_roc, name=best_name)
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax_roc.set_title(f"ROC Curve â€” {best_name} (Test Set, AUC={auc:.4f})")
fig_roc.tight_layout()
fig_roc.savefig(os.path.join(PLOTS_DIR, "roc_curve_final.png"), dpi=150)
print("    Saved â†’ reports/figures/roc_curve_final.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9. Save model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_path = os.path.join(MODELS_DIR, "scope_model.joblib")
joblib.dump(best_model, model_path)
print(f"\n[9] Model saved â†’ models/scope_model.joblib")

plt.show()
print("\nDone.")

