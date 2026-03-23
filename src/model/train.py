"""
SCOPE – Day 3: Baseline Model Training & Diagnostics
=====================================================
Pipeline:
  1. Load dataset → preprocess (via preprocess.py) → SMOTE balance
  2. Train RandomForestClassifier(n_estimators=100, random_state=42)
  3. Evaluate on X_val: classification_report + confusion matrix
  4. Visualise a single decision tree (max_depth=3)
  5. Plot feature importances (sorted)
  6. Plot ROC curve + compute AUC

Run from the project root:
    python -m src.model.train
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE

# ── allow ``python -m src.model.train`` from any cwd ──────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.model.preprocess import build_preprocessor  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# feature columns (must match preprocess.py)
NUMERICAL_FEATURES = [
    "days_since_created", "days_since_last_update", "num_versions",
    "release_velocity", "num_maintainers", "description_length",
    "stargazers_count", "forks_count", "open_issues_count",
    "subscribers_count", "contributor_count", "days_since_last_commit",
]
BINARY_FEATURES = ["has_postinstall", "license_is_standard", "has_github_repo"]
ALL_FEATURE_NAMES = NUMERICAL_FEATURES + BINARY_FEATURES   # order matches ColumnTransformer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & split
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SCOPE – Day 3 Training Script")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    sys.exit(f"[ERROR] Dataset not found at {DATASET_PATH}.\n"
             "Run src/data/build_dataset.py first.")

print(f"\n[1] Dataset loaded  shape={df.shape}")

X = df.drop(columns=["label"])
y = df["label"]

# Hold-out test set (15%) — do NOT touch until Day 8
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Validation set (~15% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print(f"    train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocess (fit on train only — no leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Fitting preprocessor on X_train …")
preprocessor = build_preprocessor()
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed   = preprocessor.transform(X_val)
X_test_transformed  = preprocessor.transform(X_test)

# Save fitted preprocessor for later inference
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
print("    Preprocessor saved → models/preprocessor.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOTE (oversample minority class in training set only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Applying SMOTE …")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)

_counts = pd.Series(y_train_res).value_counts().sort_index()
print(f"    After SMOTE  →  class 0 (Healthy): {_counts[0]}   class 1 (Suspicious): {_counts[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train RandomForest baseline
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Training RandomForestClassifier(n_estimators=100, random_state=42) …")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
print("    Training complete.")

# Save model
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_baseline.pkl"))
print("    Model saved → models/rf_baseline.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate on VALIDATION set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[5] CLASSIFICATION REPORT  (evaluated on X_val)")
print("=" * 60)

y_val_pred = rf.predict(X_val_transformed)
report = classification_report(y_val, y_val_pred, target_names=["Healthy", "Suspicious"])
print(report)

# Derive false-negative rate from the confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
# cm layout: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
print(f"  False-Negative Rate (FNR) = FN/(FN+TP) = {FN}/({FN}+{TP}) = {fnr:.4f}")
print(f"  → {fnr*100:.1f}% of suspicious packages were MISSED by the model.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Confusion matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[6] Plotting confusion matrix …")
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Healthy", "Predicted Suspicious"],
    yticklabels=["Actual Healthy",    "Actual Suspicious"],
    ax=ax_cm,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 14, "weight": "bold"},
)
ax_cm.set_title("Confusion Matrix – RF Baseline (Validation Set)", pad=12, fontsize=13)
ax_cm.set_ylabel("True Label", fontsize=11)
ax_cm.set_xlabel("Predicted Label", fontsize=11)

# Add cell-level annotations
cell_labels = [
    (0, 0, f"TN\n{TN}"),
    (0, 1, f"FP\n{FP}"),
    (1, 0, f"FN\n{FN}"),
    (1, 1, f"TP\n{TP}"),
]
for row, col, label in cell_labels:
    ax_cm.text(
        col + 0.5, row + 0.72,
        label.split("\n")[0],
        ha="center", va="center",
        fontsize=9, color="grey",
    )

fig_cm.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_rf_baseline.png")
fig_cm.savefig(cm_path, dpi=150)
print(f"    Saved → {cm_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tree visualisation – single estimator, max_depth=3
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Plotting single decision tree (estimators_[0], max_depth=3) …")
single_tree = rf.estimators_[0]

fig_tree, ax_tree = plt.subplots(figsize=(22, 9))
plot_tree(
    single_tree,
    max_depth=3,
    feature_names=ALL_FEATURE_NAMES,
    class_names=["Healthy", "Suspicious"],
    filled=True,
    rounded=True,
    impurity=True,       # shows gini value in every node
    proportion=False,    # shows raw sample counts
    fontsize=9,
    ax=ax_tree,
)
ax_tree.set_title(
    "Single Decision Tree from Random Forest  (max_depth=3)\n"
    "Colour intensity = class purity  |  Gini: 0 = pure, 0.5 = maximally impure",
    fontsize=12, pad=14,
)
fig_tree.tight_layout()
tree_path = os.path.join(PLOTS_DIR, "single_tree_depth3.png")
fig_tree.savefig(tree_path, dpi=150, bbox_inches="tight")
print(f"    Saved → {tree_path}")

# Print the root split for study
from sklearn.tree import _tree  # noqa: E402
tree_obj = single_tree.tree_
root_feature   = ALL_FEATURE_NAMES[tree_obj.feature[0]]
root_threshold = tree_obj.threshold[0]
root_gini      = tree_obj.impurity[0]
print(f"\n  ROOT NODE:")
print(f"    Feature   : {root_feature}")
print(f"    Threshold : {root_threshold:.4f}")
print(f"    Gini      : {root_gini:.4f}  (closer to 0 → purer split)")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature importances – horizontal bar chart (descending)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Plotting feature importances …")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]          # descending order
sorted_names  = [ALL_FEATURE_NAMES[i] for i in indices]
sorted_values = importances[indices]

fig_fi, ax_fi = plt.subplots(figsize=(9, 6))
colors = ["#e63946" if v >= np.median(sorted_values) else "#457b9d" for v in sorted_values]
bars = ax_fi.barh(sorted_names[::-1], sorted_values[::-1], color=colors[::-1], edgecolor="white")

for bar, val in zip(bars, sorted_values[::-1]):
    ax_fi.text(
        bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", fontsize=8,
    )

ax_fi.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
ax_fi.set_title("RF Baseline – Feature Importances (sorted descending)", fontsize=13)
ax_fi.axvline(x=0.01, color="grey", linestyle="--", linewidth=0.8, label="0.01 threshold")
ax_fi.legend(fontsize=9)
ax_fi.set_xlim(0, sorted_values.max() * 1.18)
fig_fi.tight_layout()
fi_path = os.path.join(PLOTS_DIR, "feature_importances_rf_baseline.png")
fig_fi.savefig(fi_path, dpi=150)
print(f"    Saved → {fi_path}")

print("\n  Feature importances (descending):")
for name, val in zip(sorted_names, sorted_values):
    bar_vis = "█" * int(val * 200)
    marker  = "  ← near-zero, candidate for removal" if val < 0.01 else ""
    print(f"    {name:<30} {val:.4f}  {bar_vis}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ROC curve + AUC (probability-based)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Plotting ROC curve …")
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_estimator(rf, X_val_transformed, y_val, ax=ax_roc, name="RF Baseline")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
ax_roc.set_title("ROC Curve – RF Baseline (Validation Set)", fontsize=13)
ax_roc.legend(fontsize=10)
fig_roc.tight_layout()
roc_path = os.path.join(PLOTS_DIR, "roc_curve_rf_baseline.png")
fig_roc.savefig(roc_path, dpi=150)
print(f"    Saved → {roc_path}")

y_val_proba = rf.predict_proba(X_val_transformed)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)
print(f"\n  AUC (probability-based) = {auc:.4f}")
print(f"  Interpretation: the model ranks a random suspicious package above a random healthy one "
      f"{auc*100:.1f}% of the time.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")
print(f"  False-Negative Rate : {fnr*100:.1f}%")
print(f"  ROC-AUC             : {auc:.4f}")
print(f"\nSaved plots:")
for p in [cm_path, tree_path, fi_path, roc_path]:
    print(f"  {p}")
print("\nDone.")

plt.show()
