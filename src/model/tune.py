"""
SCOPE – Day 4: Hyperparameter Tuning with RandomizedSearchCV
=============================================================
Curriculum tasks covered
─────────────────────────
Morning  – Hyperparameter study (printed explainer for every param)
Afternoon – RandomizedSearchCV(n_iter=50, cv=5, scoring='f1')
Evening  – Validation evaluation, side-by-side CM, CV scatter, overfit check

Run from the project root:
    python -m src.model.tune

Prerequisites:
    • data/processed/dataset.csv   (build_dataset.py)
    • models/preprocessor.pkl      (optional – rebuilt here if absent)
    • models/rf_baseline.pkl       (optional – rebuilt here if absent)

Outputs saved to reports/figures/:
    confusion_matrix_comparison.png
    cv_scatter_n_estimators.png

Model saved to:
    models/rf_tuned.pkl
"""

import os
import sys
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── path setup so we can run as a module from project root ────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from src.model.preprocess import build_preprocessor  # noqa: E402

DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

NUMERICAL_FEATURES = [
    "days_since_created", "days_since_last_update", "num_versions",
    "release_velocity", "num_maintainers", "description_length",
    "stargazers_count", "forks_count", "open_issues_count",
    "subscribers_count", "contributor_count", "days_since_last_commit",
]
BINARY_FEATURES    = ["has_postinstall", "license_is_standard", "has_github_repo"]
ALL_FEATURES       = NUMERICAL_FEATURES + BINARY_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0 — HYPERPARAMETER EXPLAINER  (Morning study block)
# ═════════════════════════════════════════════════════════════════════════════
EXPLAINER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║       RANDOM FOREST HYPERPARAMETER STUDY  (Morning block)                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. n_estimators  (default=100)                                              ║
║     What it does : Number of decision trees in the forest.                   ║
║     Too HIGH     : Slower training; diminishing returns; rarely overfits.    ║
║     Too LOW      : High variance; predictions unstable run-to-run.           ║
║     Sweet spot   : 100–500. Beyond 500 gains are marginal.                   ║
║                                                                              ║
║  2. max_depth  (default=None → grow until pure)                              ║
║     What it does : Maximum depth of each tree.                               ║
║     Too HIGH     : Trees memorise noise → overfitting. Train acc → 100%,    ║
║                    val acc drops.                                             ║
║     Too LOW      : Trees can't capture complex patterns → underfitting.      ║
║     Regularises  : YES – the single most powerful lever for variance.        ║
║                                                                              ║
║  3. min_samples_split  (default=2)                                           ║
║     What it does : Minimum samples needed to split an internal node.         ║
║     Too HIGH     : Nodes stop splitting early → shallower trees → underfit.  ║
║     Too LOW (=2) : Every node splits greedily → deep, overfit trees.         ║
║     Regularises  : YES – acts as a pruning mechanism.                        ║
║                                                                              ║
║  4. min_samples_leaf  (default=1)                                            ║
║     What it does : Minimum samples that must exist in a LEAF node.           ║
║     Too HIGH     : Leaves become bucket averages → smooth but biased.        ║
║     Too LOW (=1) : Single-sample leaves → extreme overfit.                   ║
║     Regularises  : YES – very effective for noisy datasets.                  ║
║                                                                              ║
║  5. max_features  (default='sqrt')                                           ║
║     What it does : How many features each tree RANDOMLY considers per split. ║
║     Too HIGH     : Trees all pick the same dominant feature → correlated     ║
║                    trees, less ensemble diversity, less variance reduction.   ║
║     Too LOW      : Each split is near-random → high bias, underfit.          ║
║     'sqrt'       : Classic RF default; good for classification.              ║
║     'log2'       : More aggressive decorrelation; try for high-dim data.     ║
║                                                                              ║
║  BIAS–VARIANCE REFRESHER                                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Bias   = error from wrong assumptions  → model too simple → underfit      ║
║  • Variance = sensitivity to training noise → model too complex → overfit    ║
║  • RF reduces variance via bagging (averaging many trees).                   ║
║  • max_depth, min_samples_leaf, min_samples_split all INCREASE bias while   ║
║    DECREASING variance (regularization). Goal: find the sweet spot.          ║
║                                                                              ║
║  MY PREDICTION (fill in before seeing results):                              ║
║  "max_depth will have the biggest impact because unconstrained trees         ║
║   memorise every minority-class quirk introduced by SMOTE synthesis."        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(EXPLAINER)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA PIPELINE  (identical to train.py for reproducibility)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1 – Data pipeline")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    sys.exit(f"[ERROR] Dataset not found at {DATASET_PATH}.\n"
             "Run src/data/build_dataset.py first.")

print(f"Dataset loaded  shape={df.shape}")

X = df.drop(columns=["label"])
y = df["label"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)
print(f"Splits — train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

preprocessor = build_preprocessor()
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed   = preprocessor.transform(X_val)
X_test_transformed  = preprocessor.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)
_c = pd.Series(y_train_res).value_counts().sort_index()
print(f"After SMOTE — class 0: {_c[0]}  class 1: {_c[1]}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD (OR REBUILD) BASELINE FOR COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2 – Baseline model (Day 3 RF, n_estimators=100)")
print("=" * 60)

baseline_path = os.path.join(MODELS_DIR, "rf_baseline.pkl")
if os.path.exists(baseline_path):
    rf_baseline = joblib.load(baseline_path)
    print("Loaded existing baseline → models/rf_baseline.pkl")
else:
    print("Baseline not found — training it now …")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train_res, y_train_res)
    joblib.dump(rf_baseline, baseline_path)
    print("Baseline trained and saved.")

y_base_val  = rf_baseline.predict(X_val_transformed)
base_f1     = f1_score(y_val, y_base_val)
base_cm     = confusion_matrix(y_val, y_base_val)
base_auc    = roc_auc_score(y_val, rf_baseline.predict_proba(X_val_transformed)[:, 1])

print(f"\nBaseline  val-F1={base_f1:.4f}   AUC={base_auc:.4f}")
print(classification_report(y_val, y_base_val, target_names=["Healthy", "Suspicious"]))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RANDOMIZED SEARCH  (Afternoon block)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 3 – RandomizedSearchCV  (n_iter=50, cv=5, scoring='f1')")
print("=" * 60)

# ── What cv=5 means ──────────────────────────────────────────────────────────
print("""
HOW cv=5 WORKS:
  The resampled training set (X_train_res) is split into 5 equal folds.
  For each of the 50 random hyperparameter combos:
    • Fold 1 as held-out  → train on folds 2–5 → compute F1 on fold 1
    • Fold 2 as held-out  → train on folds 1,3–5 → compute F1 on fold 2
    • … (repeat 5 times)
    • mean_test_score = average of those 5 F1 values
  Total model fits: 50 combos × 5 folds = 250 fits.
  We pick the combo with the highest mean F1.
""")

# ── Parameter distribution ───────────────────────────────────────────────────
param_dist = {
    # More trees = more stable, rarely hurts; diminishing returns past 500
    "n_estimators": randint(100, 600),

    # None = full trees (overfit); restrict to 5–30 for regularisation
    "max_depth": [None, 5, 10, 15, 20, 25, 30],

    # How many samples needed before splitting a node (pruning lever)
    "min_samples_split": randint(2, 20),

    # How many samples must remain in a leaf (very effective regulariser)
    "min_samples_leaf": randint(1, 10),

    # Feature subset per split; sqrt vs log2 decorrelates trees differently
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}

print("Parameter search space:")
for k, v in param_dist.items():
    print(f"  {k:<22} {v}")

rf_base_for_search = RandomForestClassifier(random_state=42, n_jobs=1)

search = RandomizedSearchCV(
    estimator  = rf_base_for_search,
    param_distributions = param_dist,
    n_iter     = 50,          # 50 random combinations
    cv         = 5,           # 5-fold cross-validation
    scoring    = "f1",        # optimise for F1 on the minority class
    n_jobs     = -1,          # use all CPU cores
    random_state = 42,
    verbose    = 2,           # show progress for each fit
    refit      = True,        # refit best model on full X_train_res after search
)

print(f"\nStarting search — 50 × 5 = 250 fits. This may take 15–30 min …\n")
t0 = time.time()
search.fit(X_train_res, y_train_res)
elapsed = time.time() - t0

print(f"\nSearch complete in {elapsed/60:.1f} min.")
print(f"\nbest_params_  = {search.best_params_}")
print(f"best_score_   = {search.best_score_:.4f}  (mean CV F1 on training folds)")

# Compare CV score to naive baseline
print(f"\nNaive baseline val-F1 : {base_f1:.4f}")
print(f"Tuned   CV   mean-F1  : {search.best_score_:.4f}   "
      f"({'↑ IMPROVED' if search.best_score_ > base_f1 else '↓ degraded'} vs baseline)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VALIDATION EVALUATION OF BEST ESTIMATOR  (Evening block)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4 – Evaluation on X_val (best_estimator_)")
print("=" * 60)

rf_tuned = search.best_estimator_
joblib.dump(rf_tuned, os.path.join(MODELS_DIR, "rf_tuned.pkl"))
print("Tuned model saved → models/rf_tuned.pkl")

y_tuned_val  = rf_tuned.predict(X_val_transformed)
tuned_f1     = f1_score(y_val, y_tuned_val)
tuned_cm     = confusion_matrix(y_val, y_tuned_val)
tuned_auc    = roc_auc_score(y_val, rf_tuned.predict_proba(X_val_transformed)[:, 1])

print(f"\nTuned model  val-F1={tuned_f1:.4f}   AUC={tuned_auc:.4f}")
print(classification_report(y_val, y_tuned_val, target_names=["Healthy", "Suspicious"]))

# ── metric delta table ────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score  # noqa: E402
metrics = {
    "Precision (Suspicious)": (
        precision_score(y_val, y_base_val),
        precision_score(y_val, y_tuned_val),
    ),
    "Recall (Suspicious)": (
        recall_score(y_val, y_base_val),
        recall_score(y_val, y_tuned_val),
    ),
    "F1 (Suspicious)": (base_f1, tuned_f1),
    "ROC-AUC":         (base_auc, tuned_auc),
}
print("\n  Metric comparison:")
print(f"  {'Metric':<28} {'Baseline':>10}  {'Tuned':>10}  {'Δ':>8}")
print("  " + "─" * 62)
for name, (b, t) in metrics.items():
    delta = t - b
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"  {name:<28} {b:>10.4f}  {t:>10.4f}  {arrow} {abs(delta):.4f}")

# ── Side-by-side confusion matrices ──────────────────────────────────────────
print("\nPlotting side-by-side confusion matrices …")
fig_cmp, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, cm_data, title, y_pred in [
    (axes[0], base_cm,  "Baseline RF (n_estimators=100,\ndefault params)", y_base_val),
    (axes[1], tuned_cm, f"Tuned RF (best_params_)\nval-F1={tuned_f1:.3f}", y_tuned_val),
]:
    sns.heatmap(
        cm_data, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred Healthy", "Pred Suspicious"],
        yticklabels=["Actual Healthy", "Actual Suspicious"],
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)

    TN, FP, FN, TP = cm_data.ravel()
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    ax.text(
        0.5, -0.18,
        f"TN={TN}  FP={FP}  FN={FN}  TP={TP}   FNR={fnr:.1%}",
        ha="center", transform=ax.transAxes, fontsize=9, color="#333333",
    )

fig_cmp.suptitle(
    "Confusion Matrix Comparison — Baseline vs. Tuned RF  (Validation Set)",
    fontsize=13, y=1.02,
)
fig_cmp.tight_layout()
cmp_path = os.path.join(PLOTS_DIR, "confusion_matrix_comparison.png")
fig_cmp.savefig(cmp_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {cmp_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CV scatter — n_estimators vs mean_test_score
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlotting CV results scatter (n_estimators vs mean_test_score) …")
cv_df = pd.DataFrame(search.cv_results_)

fig_sc, ax_sc = plt.subplots(figsize=(9, 5))
sc = ax_sc.scatter(
    cv_df["param_n_estimators"],
    cv_df["mean_test_score"],
    c=cv_df["mean_test_score"],
    cmap="RdYlGn",
    s=70,
    edgecolors="white",
    linewidths=0.4,
    alpha=0.85,
)
plt.colorbar(sc, ax=ax_sc, label="mean CV F1")

# Highlight the best point
best_idx = cv_df["mean_test_score"].idxmax()
ax_sc.scatter(
    cv_df.loc[best_idx, "param_n_estimators"],
    cv_df.loc[best_idx, "mean_test_score"],
    marker="*", s=280, color="gold", edgecolors="black", linewidths=0.8,
    zorder=5, label=f"Best  (n_est={cv_df.loc[best_idx,'param_n_estimators']}, "
                    f"F1={cv_df.loc[best_idx,'mean_test_score']:.4f})",
)

ax_sc.set_xlabel("n_estimators", fontsize=12)
ax_sc.set_ylabel("Mean CV F1 (5-fold)", fontsize=12)
ax_sc.set_title("RandomizedSearchCV — n_estimators vs. Mean CV F1", fontsize=13)
ax_sc.legend(fontsize=9)
ax_sc.grid(True, alpha=0.3)
fig_sc.tight_layout()
sc_path = os.path.join(PLOTS_DIR, "cv_scatter_n_estimators.png")
fig_sc.savefig(sc_path, dpi=150)
print(f"  Saved → {sc_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Overfit check — train vs val gap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 – Overfit check (train vs val F1)")
print("=" * 60)

y_tuned_train = rf_tuned.predict(X_train_res)
train_f1_tuned = f1_score(y_train_res, y_tuned_train)

y_base_train  = rf_baseline.predict(X_train_res)
train_f1_base = f1_score(y_train_res, y_base_train)

print(f"\n  {'Model':<20} {'Train F1':>10}  {'Val F1':>10}  {'Gap (↑=overfit)':>16}")
print("  " + "─" * 62)
for name, train_f1, val_f1 in [
    ("Baseline RF",  train_f1_base,  base_f1),
    ("Tuned RF",     train_f1_tuned, tuned_f1),
]:
    gap = train_f1 - val_f1
    verdict = "⚠ OVERFIT"  if gap > 0.10 else \
              "✓ mild gap" if gap > 0.03 else \
              "✓ well-fit"
    print(f"  {name:<20} {train_f1:>10.4f}  {val_f1:>10.4f}  {gap:>+10.4f}  {verdict}")

print("""
  Interpretation guide:
    gap > 0.10 → strong overfitting; increase min_samples_leaf or reduce max_depth
    gap 0.03–0.10 → mild overfitting; acceptable for tree ensembles
    gap < 0.03 → model generalises well; may still have headroom to add capacity
""")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Search time          : {elapsed/60:.1f} min")
print(f"  Best params          : {search.best_params_}")
print(f"  Best CV F1 (train)   : {search.best_score_:.4f}")
print(f"  Val F1   baseline    : {base_f1:.4f}")
print(f"  Val F1   tuned       : {tuned_f1:.4f}   "
      f"({'↑' if tuned_f1 > base_f1 else '↓'} {abs(tuned_f1-base_f1):.4f})")
print(f"  Val AUC  tuned       : {tuned_auc:.4f}")
print(f"\n  Plots saved:")
for p in [cmp_path, sc_path]:
    print(f"    {p}")
print("\nDone. Next step → Day 5 (threshold tuning / final model evaluation).")

plt.show()
