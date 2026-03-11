# SCOPE - Security Check for Open-source Package Ecosystems

**ML-powered supply chain security scanner for npm packages.**

SCOPE detects malicious, typosquatted, and suspicious npm packages before you install them, using a Random Forest classifier trained on package metadata, enriched with GitHub signals, and explained via SHAP values. It ships as a CLI tool, a web UI, and a GitHub Action that blocks risky dependencies at the PR level.

---

## The Problem

`npm audit` only catches known CVEs. It has no opinion on a brand-new package named `react-domm` published yesterday by an anonymous maintainer with a postinstall script. Supply chain attacks increasingly exploit this blind spot - SCOPE closes it.

---

## How It Works

```
Package name
     |
     v
npm Registry API ──> raw metadata (versions, maintainers, scripts, license...)
GitHub API       ──> repo signals (stars, contributors, last commit, issues...)
     |
     v
Feature Engineering (15+ signals computed from raw data)
     |
     v
Random Forest Classifier (trained on 600 labelled packages)
     |
     v
SHAP Explainer (per-prediction feature attribution)
     |
     v
Health Score + Human-readable explanation
```

The model flags packages based on signals like: days since first publish, release velocity, contributor count, presence of a `postinstall` script, download-to-star ratio, and more. Every prediction comes with a ranked explanation of *why* — not just a score.

---

## Features

- **CLI tool** — scan any package or your entire `package.json` in seconds
- **Web UI** — browser-based scanner with animated risk gauge and factor breakdown
- **GitHub Action** — blocks PRs that introduce high-risk dependencies
- **Typosquat detection** — flags packages with names similar to popular libraries
- **SHAP explanations** — every score shows which features drove the prediction
- **Carry-forward scheduling** — 18-day interactive build checklist included

---

## Project Structure

```
SCOPE/
├── data/
│   ├── raw/                    # Raw JSON from npm + GitHub APIs
│   └── processed/
│       └── dataset.csv         # Labelled training dataset
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── npm_fetcher.py      # npm Registry API client
│   │   ├── github_fetcher.py   # GitHub API client
│   │   └── feature_engineer.py # 15+ feature computations
│   ├── model/
│   │   ├── preprocess.py       # sklearn Pipeline + SMOTE
│   │   ├── train.py            # Random Forest training + tuning
│   │   ├── evaluate.py         # Metrics, ROC, confusion matrix
│   │   └── explain.py          # SHAP explainer + health score text
│   ├── cli/
│   │   ├── scope.py         # CLI entrypoint (argparse)
│   │   └── output.py           # Rich terminal formatting
│   └── api/
│       └── main.py             # FastAPI backend
├── models/
│   ├── scope_model.pkl      # Trained Random Forest
│   └── preprocessor.pkl        # Fitted sklearn Pipeline
├── tests/
│   └── test_feature_engineer.py
├── .github/
│   └── workflows/
│       └── scope-scan.yml   # GitHub Action
├── frontend/
│   └── index.html              # Web UI (vanilla JS)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- A GitHub personal access token (for GitHub API enrichment)

### Install

```bash
git clone https://github.com/yourusername/npm
pip install -r requirements.txt
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Add your GitHub token:
# GITHUB_TOKEN=ghp_your_token_here
```

### Scan a package

```bash
scope check lodash
scope check react-domm
```

### Scan your project's dependencies

```bash
scope batch package.json
scope batch package.json --fail-on-high   # exit code 1 if any package > 80% risk
```

### Output options

```bash
scope check axios --json    # machine-readable JSON output
scope --version
scope --help
```

---

## Sample Output

```
Package   : react-domm
Version   : 1.0.0  (published 1 day ago)
Risk Score: ████████████████░░░░  94%

STATUS: HIGH RISK — DO NOT INSTALL

Top Risk Factors:
  + 42%   Has postinstall script
  + 31%   Published less than 7 days ago
  + 18%   Zero GitHub contributors
  -  8%   Has repository link (partial trust)

Did you mean: react-dom?
This package name is 1 edit away from a popular library.
```

---

## GitHub Action

Add to your repo to automatically scan dependencies on every pull request that modifies `package.json`.

```yaml
# .github/workflows/scope.yml

name: SCOPE

on:
  pull_request:
    paths:
      - 'package.json'
      - 'package-lock.json'

jobs:
  scope-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install SCOPE
        run: pip install -r requirements.txt && pip install -e .

      - name: Run SCOPE Scan
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: scope batch package.json --json --fail-on-high
```

When a pull request adds a package that scores above 80% risk, the Action fails and blocks the merge. Results are posted as a PR comment.

---

## Web UI

Start the API server:

```bash
uvicorn src.api.main:app --reload
```

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend && python -m http.server 3000
```

Visit `http://localhost:3000`.

---

## ML Pipeline

### Dataset

- **600 packages** total — 400 healthy (top npm packages by downloads), 200 suspicious (confirmed malicious packages from public advisories + synthetic typosquats)
- **15 features** engineered from npm and GitHub metadata
- **Class imbalance** addressed with SMOTE (Synthetic Minority Over-sampling Technique)

### Features

| Feature | Type | Signal |
|---|---|---|
| `days_since_created` | Numerical | Brand-new packages are higher risk |
| `days_since_last_update` | Numerical | Abandoned packages are higher risk |
| `num_versions` | Numerical | Version count |
| `release_velocity` | Numerical | Versions per day — burst releases are suspicious |
| `num_maintainers` | Numerical | Single-maintainer concentration risk |
| `has_postinstall` | Binary | Most common malware execution vector |
| `has_github_repo` | Binary | Missing repo link is a red flag |
| `github_stars` | Numerical | Legitimacy signal |
| `github_contributors` | Numerical | Community health signal |
| `download_to_star_ratio` | Numerical | High downloads vs low stars = anomalous |
| `open_issues_ratio` | Numerical | Maintenance signal |
| `days_since_last_commit` | Numerical | Active development signal |
| `description_length` | Numerical | Thin descriptions correlate with malicious packages |
| `num_keywords` | Numerical | Keyword stuffing signal |
| `license_is_standard` | Binary | No/unusual license is a flag |

### Model

- **Algorithm:** Random Forest Classifier
- **Tuning:** RandomizedSearchCV with 5-fold cross-validation
- **Explainability:** SHAP TreeExplainer (per-prediction feature attribution)
- **Evaluation:** F1 score on held-out test set (80/20 split, stratified)

### Model Performance

| Metric | Healthy | Suspicious |
|---|---|---|
| Precision | — | — |
| Recall | — | — |
| F1 Score | — | — |
| ROC-AUC | — | — |

*Fill in after training. Target: F1 ≥ 0.82 on the suspicious class.*

---

## API Reference

The FastAPI backend exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Analyze a single package |
| `/batch` | POST | Analyze up to 20 packages |
| `/health` | GET | Server health check |
| `/docs` | GET | Auto-generated Swagger UI |

### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"package_name": "react-domm"}'
```

### Example response

```json
{
  "package_name": "react-domm",
  "risk_score": 0.94,
  "risk_level": "HIGH",
  "factors": [
    { "feature": "has_postinstall", "contribution": 0.42, "direction": "risk" },
    { "feature": "days_since_created", "contribution": 0.31, "direction": "risk" },
    { "feature": "github_contributors", "contribution": 0.18, "direction": "risk" }
  ],
  "recommendation": "Do not install. Verify this package manually.",
  "suggested_alternative": "react-dom"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data collection | npm Registry API, GitHub REST API |
| Data processing | pandas, numpy |
| ML — classification | scikit-learn RandomForestClassifier |
| ML — imbalance | imbalanced-learn (SMOTE) |
| ML — explainability | SHAP (TreeExplainer) |
| ML — tuning | RandomizedSearchCV |
| CLI | Python argparse, rich |
| API | FastAPI, Pydantic, uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| CI/CD | GitHub Actions |
| Deployment | Railway (API), Vercel (frontend) |

---

## Build Checklist

This project was built following an 18-day structured plan. The interactive checklist (with carry-forward overflow logic for missed tasks) is available as a React component:

→ 

---

## Limitations

- Metadata-only analysis. SCOPE does not download or execute package code.
- A sophisticated attacker who mimics the metadata patterns of healthy packages (fake stars, sleeper accounts, curated history) can reduce SCOPE's detection rate.
- The model is only as good as its training data. New attack patterns not represented in the dataset will not be reliably detected.
- Not a substitute for manual review of critical dependencies.

---

## Roadmap

- NLP layer: semantic coherence between package name, description, and exports
- Anomaly detection via Isolation Forest (zero-day pattern detection)
- Graph analysis: dependency graph centrality scoring
- AST-level static analysis for obfuscated code patterns
- CV-based logo similarity detection for visual typosquatting


---

## Disclaimer

SCOPE is a heuristic tool. A HIGH RISK score is a signal to investigate, not a definitive verdict. A LOW RISK score does not guarantee a package is safe. Always apply judgment before installing dependencies in production systems.
