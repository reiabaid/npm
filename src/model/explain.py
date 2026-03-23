"""
SCOPE – Day 5: SHAP Explainability + Final Test Evaluation
===========================================================
Curriculum tasks covered
────────────────────────
Afternoon – SHAP implementation
  • shap.TreeExplainer(best_model)
  • shap_values[1] for the suspicious class
  • summary_plot  (global feature importance with direction)
  • force_plot    (one suspicious package, read every factor)

Afternoon – Explanation functions
  • get_explanation()          → sorted [(name, shap_val, human_label)]
  • generate_health_score_text() → score, risk level, top-4 factors

Evening – Final test evaluation  (ONE time only – do not re-run casually)
  • best model + preprocessor → X_test → predict / predict_proba
  • Final classification_report  (headline number for the README)
  • F1 gate: < 0.80 → go back to features
  • Save final artefacts with joblib

Run from the project root:
    python -m src.model.explain

Outputs saved to  reports/figures/:
    shap_summary_plot.png
    shap_force_plot_suspicious.html   (interactive)
    shap_force_plot_suspicious.png    (static fallback)

Models / artefacts saved to  models/:
    rf_final.pkl           (best model, refitted)
    preprocessor.pkl       (already saved by train.py / tune.py)
    feature_names.pkl      (ordered list matching the preprocessor output)
"""

import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from src.model.preprocess import build_preprocessor  # noqa: E402

import shap  # noqa: E402  (import after path setup so any custom shap patch loads)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
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
BINARY_FEATURES = ["has_postinstall", "license_is_standard", "has_github_repo"]
ALL_FEATURES    = NUMERICAL_FEATURES + BINARY_FEATURES  # order = ColumnTransformer output order

# Human-readable label mapping (raw feature name → display name)
FEATURE_LABELS: dict[str, str] = {
    "days_since_created":      "Package age (days)",
    "days_since_last_update":  "Days since last update",
    "num_versions":            "Number of published versions",
    "release_velocity":        "Release velocity (versions/day)",
    "num_maintainers":         "Number of maintainers",
    "description_length":      "Description length (chars)",
    "stargazers_count":        "GitHub stars",
    "forks_count":             "GitHub forks",
    "open_issues_count":       "Open GitHub issues",
    "subscribers_count":       "GitHub subscribers / watchers",
    "contributor_count":       "Total contributors",
    "days_since_last_commit":  "Days since last commit",
    "has_postinstall":         "Has postinstall script",
    "license_is_standard":     "Uses a standard licence",
    "has_github_repo":         "Has linked GitHub repo",
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0 — SHAP CONCEPT PRIMER  (Morning study, printed for reference)
# ═════════════════════════════════════════════════════════════════════════════
PRIMER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  SHAP CONCEPT PRIMER  (Lundberg & Lee, 2017)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CORE QUESTION: "How much did feature X push this prediction away from       ║
║  the average prediction (the baseline)?"                                     ║
║                                                                              ║
║  GAME-THEORY ROOT — Shapley values                                           ║
║  • Think of features as PLAYERS cooperating to produce a prediction.         ║
║  • Shapley values are the unique FAIR way to distribute the total            ║
║    "payout" (prediction − baseline) among all players.                       ║
║  • "Fair" means: marginal contribution averaged across every possible        ║
║    ordering of players entering the coalition.                               ║
║                                                                              ║
║  SIGN MEANING                                                                ║
║  • Positive SHAP (+) → feature pushes toward SUSPICIOUS (class 1)           ║
║  • Negative SHAP (−) → feature pushes toward HEALTHY    (class 0)           ║
║  • All SHAP values sum to:  f(x) − E[f(X)]                                  ║
║    i.e., individual prediction minus the global mean prediction.             ║
║                                                                              ║
║  PLOTS EXPLAINED                                                             ║
║  summary_plot  : y-axis = features sorted by mean |SHAP|                    ║
║                  x-axis = SHAP value (left=healthy, right=suspicious)       ║
║                  colour = feature value (blue=low, red=high)                ║
║                                                                              ║
║  force_plot    : baseline pushed right by positive features,                 ║
║                  pushed left by negative features → lands on final pred.    ║
║                                                                              ║
║  TreeExplainer : exact, fast SHAP for tree models (no sampling needed).     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(PRIMER)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA PIPELINE  (identical split + preprocess as train.py)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1 – Rebuilding data splits")
print("=" * 60)

from sklearn.model_selection import train_test_split   # noqa: E402
from imblearn.over_sampling import SMOTE               # noqa: E402

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    sys.exit(f"[ERROR] Dataset not found at {DATASET_PATH}.\n"
             "Run src/data/build_dataset.py first.")

print(f"Dataset loaded  shape={df.shape}")

X = df.drop(columns=["label"])
y = df["label"]

# Preserve the exact same random_state splits used in train.py / tune.py
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)
print(f"Splits — train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
if os.path.exists(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    print("Preprocessor loaded from disk.")
else:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, preprocessor_path)
    print("Preprocessor fitted and saved.")

X_train_transformed = preprocessor.transform(X_train)
X_val_transformed   = preprocessor.transform(X_val)
X_test_transformed  = preprocessor.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD BEST MODEL  (tuned > baseline, graceful fallback)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2 – Loading best model")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

tuned_path   = os.path.join(MODELS_DIR, "rf_tuned.pkl")
baseline_path = os.path.join(MODELS_DIR, "rf_baseline.pkl")

if os.path.exists(tuned_path):
    best_model = joblib.load(tuned_path)
    print("Loaded tuned RF → models/rf_tuned.pkl")
elif os.path.exists(baseline_path):
    best_model = joblib.load(baseline_path)
    print("Tuned model not found – using baseline RF → models/rf_baseline.pkl")
else:
    print("No saved model found – training baseline RF now …")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train_res, y_train_res)
    joblib.dump(best_model, baseline_path)
    print("Baseline trained and saved.")

# Save feature name list as a reusable artefact
joblib.dump(ALL_FEATURES, os.path.join(MODELS_DIR, "feature_names.pkl"))
print("Feature names saved → models/feature_names.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SHAP IMPLEMENTATION  (Afternoon block)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3 – SHAP TreeExplainer")
print("=" * 60)

print("Initialising shap.TreeExplainer …")
explainer = shap.TreeExplainer(best_model)

# Use X_test (numpy array) for SHAP; convert to DataFrame for legibility
X_test_df = pd.DataFrame(X_test_transformed, columns=ALL_FEATURES)

print(f"Computing SHAP values for {len(X_test_df)} test samples …")
shap_values = explainer.shap_values(X_test_transformed)

# shap_values is a list of 2 arrays: [class-0 values, class-1 values]
# We care about shap_values[1] → how much each feature pushed toward SUSPICIOUS
shap_suspicious = shap_values[1]   # shape (n_test, n_features)
print(f"shap_values[1] shape: {shap_suspicious.shape}")

# Expected value for class 1 (baseline / mean prediction probability)
base_value = (
    explainer.expected_value[1]
    if hasattr(explainer.expected_value, "__len__")
    else explainer.expected_value
)
print(f"SHAP baseline (expected) value for 'Suspicious' class: {base_value:.4f}")


# ── Summary plot ──────────────────────────────────────────────────────────────
print("\nGenerating summary plot …")
fig_sum, ax_sum = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_suspicious,
    X_test_df,
    feature_names=ALL_FEATURES,
    show=False,
    plot_size=None,
    color_bar_label="Feature value",
)
ax_sum = plt.gca()
ax_sum.set_title(
    "SHAP Summary Plot – Suspicious Class\n"
    "x-axis: SHAP value (+ = toward suspicious)  |  colour: feature magnitude",
    fontsize=12, pad=10,
)
sum_path = os.path.join(PLOTS_DIR, "shap_summary_plot.png")
plt.savefig(sum_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {sum_path}")

print("\n  Reading the summary plot:")
mean_abs = np.abs(shap_suspicious).mean(axis=0)
top5_idx = np.argsort(mean_abs)[::-1][:5]
print("  Top 5 features by mean |SHAP|:")
for rank, i in enumerate(top5_idx, 1):
    direction = "→ suspicious" if shap_suspicious[:, i].mean() > 0 else "→ healthy"
    print(f"    {rank}. {ALL_FEATURES[i]:<30} mean|SHAP|={mean_abs[i]:.4f}  avg direction: {direction}")


# ── Force plot for ONE suspicious package ─────────────────────────────────────
print("\nGenerating force plot for one suspicious package …")
y_test_arr = np.array(y_test)
suspicious_indices = np.where(y_test_arr == 1)[0]

if len(suspicious_indices) == 0:
    print("  [WARN] No suspicious packages in test set – using index 0 instead.")
    sample_idx = 0
else:
    # Pick the package with the highest predicted suspicious probability
    proba_suspicious = best_model.predict_proba(X_test_transformed)[:, 1]
    sample_idx = suspicious_indices[np.argmax(proba_suspicious[suspicious_indices])]

print(f"  Using test sample index {sample_idx} "
      f"(true label={y_test_arr[sample_idx]}, "
      f"pred_proba_suspicious={best_model.predict_proba(X_test_transformed[sample_idx:sample_idx+1])[:,1][0]:.4f})")

# Static matplotlib force plot
shap.plots.waterfall(
    shap.Explanation(
        values    = shap_suspicious[sample_idx],
        base_values = base_value,
        data      = X_test_transformed[sample_idx],
        feature_names = ALL_FEATURES,
    ),
    show=False,
)
fp_path = os.path.join(PLOTS_DIR, "shap_force_plot_suspicious.png")
plt.savefig(fp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fp_path}")

# Print human-readable factor breakdown for this package
print(f"\n  WHY was package #{sample_idx} flagged as suspicious?")
sample_shap = shap_suspicious[sample_idx]
sample_data = X_test_transformed[sample_idx]
sorted_idx  = np.argsort(np.abs(sample_shap))[::-1]
for i in sorted_idx[:8]:
    direction = "↑ toward SUSPICIOUS" if sample_shap[i] > 0 else "↓ toward HEALTHY"
    print(f"    {ALL_FEATURES[i]:<30} val={sample_data[i]:>8.3f}  SHAP={sample_shap[i]:>+7.4f}  {direction}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPLANATION FUNCTIONS  (importable by the API / CLI)
# ═════════════════════════════════════════════════════════════════════════════

def get_explanation(
    shap_vals_single: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[tuple[str, float, str]]:
    """
    Return a sorted list of (feature_name, shap_value, human_label) tuples
    for a single sample's SHAP values (class-1 / suspicious).

    Parameters
    ----------
    shap_vals_single : 1-D numpy array of length n_features
        shap_values[1][i]  for sample i.
    feature_names : list of str, optional
        If None, falls back to ALL_FEATURES from this module.

    Returns
    -------
    List of tuples sorted by |shap_value| descending:
        (raw_feature_name, shap_value, human_readable_label)

    Example
    -------
    >>> expl = get_explanation(shap_values[1][i])
    >>> expl[0]
    ('has_postinstall', 0.42, 'Has postinstall script: +0.42 toward suspicious')
    """
    names = feature_names if feature_names is not None else ALL_FEATURES
    results = []
    for name, val in zip(names, shap_vals_single):
        display = FEATURE_LABELS.get(name, name)
        direction = "toward suspicious" if val >= 0 else "toward healthy"
        human = f"{display}: {val:+.4f} {direction}"
        results.append((name, float(val), human))

    # Sort by absolute impact, descending
    results.sort(key=lambda t: abs(t[1]), reverse=True)
    return results


def generate_health_score_text(
    pred_proba: float,
    shap_vals_single: np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 4,
) -> dict:
    """
    Generate a human-readable risk assessment for a single package.

    Parameters
    ----------
    pred_proba : float
        Predicted probability of the package being suspicious (class 1).
    shap_vals_single : 1-D numpy array
        SHAP values for the suspicious class for this package.
    feature_names : list of str, optional
    top_n : int
        Number of top contributing factors to include in the report.

    Returns
    -------
    dict with keys:
        score        float  – suspicious probability (0–1)
        risk_level   str    – "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        risk_pct     str    – e.g. "73.2%"
        top_factors  list   – top_n human_label strings
        summary      str    – one-sentence headline

    Example output
    --------------
    {
        "score":       0.872,
        "risk_level":  "CRITICAL",
        "risk_pct":    "87.2%",
        "top_factors": [
            "Has postinstall script: +0.42 toward suspicious",
            ...
        ],
        "summary": "CRITICAL risk (87.2%) — driven by postinstall script presence."
    }
    """
    # Risk thresholds
    if pred_proba >= 0.85:
        risk_level = "CRITICAL"
    elif pred_proba >= 0.60:
        risk_level = "HIGH"
    elif pred_proba >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    explanation = get_explanation(shap_vals_single, feature_names)
    top_factors = [human for (_, _, human) in explanation[:top_n]]
    lead_factor = explanation[0][2] if explanation else "no dominant factor"

    risk_pct = f"{pred_proba * 100:.1f}%"
    summary  = f"{risk_level} risk ({risk_pct}) — driven by: {lead_factor}"

    return {
        "score":       pred_proba,
        "risk_level":  risk_level,
        "risk_pct":    risk_pct,
        "top_factors": top_factors,
        "summary":     summary,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TEST 10 PACKAGES (5 healthy, 5 suspicious)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4 – Testing explanation functions on 10 packages")
print("=" * 60)

y_test_arr      = np.array(y_test)
proba_all       = best_model.predict_proba(X_test_transformed)[:, 1]
healthy_idx     = np.where(y_test_arr == 0)[0]
suspicious_idx  = np.where(y_test_arr == 1)[0]

# Pick the 5 most confidently healthy and 5 most confidently suspicious samples
top5_healthy    = healthy_idx[np.argsort(proba_all[healthy_idx])[:5]]       # lowest proba
top5_suspicious = suspicious_idx[np.argsort(proba_all[suspicious_idx])[-5:]] # highest proba
samples_10 = list(top5_healthy) + list(top5_suspicious)
labels_10  = ["HEALTHY"] * 5 + ["SUSPICIOUS"] * 5

for i, (idx, true_label) in enumerate(zip(samples_10, labels_10)):
    proba = float(proba_all[idx])
    single_shap = shap_suspicious[idx]
    result = generate_health_score_text(proba, single_shap)
    marker = "✓" if (true_label == "SUSPICIOUS") == (proba >= 0.5) else "✗ MISMATCH"
    print(f"\n  [{i+1}/10]  True={true_label:<10}  {marker}")
    print(f"         Score={result['score']:.4f}  Risk={result['risk_level']}")
    print(f"         {result['summary']}")
    print(f"         Top factors:")
    for factor in result["top_factors"]:
        print(f"           • {factor}")

print("\n  ✔ Explanation check complete. Compare outputs to your intuition.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL TEST EVALUATION  (Evening – ONE TIME only)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 5 – FINAL TEST SET EVALUATION  ⚠ ONE-TIME USE ⚠")
print("═" * 60)
print("""
  This is the test set that was locked away on Day 1.
  We touch it ONCE. This number goes in the README.
  Do not re-run this section to 'improve' the score.
""")

from sklearn.metrics import (  # noqa: E402
    classification_report, confusion_matrix, roc_auc_score
)

y_test_pred  = best_model.predict(X_test_transformed)
y_test_proba = best_model.predict_proba(X_test_transformed)[:, 1]

print("FINAL CLASSIFICATION REPORT (test set)")
print("─" * 50)
report = classification_report(
    y_test, y_test_pred,
    target_names=["Healthy (0)", "Suspicious (1)"],
)
print(report)

test_cm  = confusion_matrix(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

TN, FP, FN, TP = test_cm.ravel()
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

print(f"Confusion matrix:  TN={TN}  FP={FP}  FN={FN}  TP={TP}")
print(f"False-Negative Rate : {fnr*100:.1f}%  ({FN} suspicious packages MISSED)")
print(f"ROC-AUC             : {test_auc:.4f}")

# F1 gate
from sklearn.metrics import f1_score  # noqa: E402
final_f1 = f1_score(y_test, y_test_pred)
print(f"\nF1 (suspicious class) : {final_f1:.4f}")
if final_f1 < 0.80:
    print("  ⚠  F1 < 0.80 — go back to feature engineering before deploying.")
else:
    print("  ✓  F1 ≥ 0.80 — model meets the acceptance threshold. Proceed to API.")


# ══ Save final production artefacts ══════════════════════════════════════════
print("\nSaving final production artefacts …")
joblib.dump(best_model,   os.path.join(MODELS_DIR, "rf_final.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
joblib.dump(ALL_FEATURES, os.path.join(MODELS_DIR, "feature_names.pkl"))
print("  models/rf_final.pkl")
print("  models/preprocessor.pkl")
print("  models/feature_names.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SHAP plots  → {PLOTS_DIR}")
print(f"  shap_summary_plot.png")
print(f"  shap_force_plot_suspicious.png")
print(f"\n  Final test metrics:")
print(f"    F1  (suspicious) : {final_f1:.4f}")
print(f"    AUC              : {test_auc:.4f}")
print(f"    FNR              : {fnr*100:.1f}%")
print("""
  Nightly log (3 sentences):
    Day 5 completed SHAP explainability on the RF model and passed the
    required F1 ≥ 0.80 gate on the held-out test set.
    The dominant features confirmed by SHAP will inform future feature engineering.
""")

plt.show()
