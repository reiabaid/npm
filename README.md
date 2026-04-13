# SCOPE - Security Check for Open-source Package Ecosystems

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/mit)
[![SCOPE Scan](https://github.com/reiabaid/npm/actions/workflows/scope.yml/badge.svg)](https://github.com/reiabaid/npm/actions/workflows/scope.yml)

SCOPE is an explainable npm dependency scanner that flags risky packages in the CLI, web UI, and GitHub Actions.

## Demo

Add a screenshot or GIF here.

## How It Works

1. You scan a package or `package.json`.
2. SCOPE reads public npm and GitHub data about that package.
3. It turns that data into a small set of risk signals.
4. A trained model scores the package from 0 to 1.
5. The explanation layer shows why the score is high or low.

## Quick Start

```bash
git clone https://github.com/reiabaid/npm.git
cd npm
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
uvicorn src.api.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## CLI Examples

```bash
scope check lodash
scope check react-domm
scope batch package.json --json --fail-on-high
```

`--fail-on-high` exits with code `1` when any package score is above `0.80`.

## GitHub Action

Use [`.github/workflows/scope.yml`](.github/workflows/scope.yml) in a repo with npm dependencies. It runs on pull requests that change `package.json` or `package-lock.json`, posts a PR comment, and fails the check if a package is too risky.

## Model Performance

- F1 score, suspicious class: `0.88`
- ROC-AUC: `0.9886`

## Main Features

- CLI scanning for single packages or project files
- React web UI for interactive scoring
- GitHub Action for PR blocking
- SHAP-style explanations for each score
- npm and GitHub metadata enrichment

## Tech Stack

- Python, FastAPI, Uvicorn
- scikit-learn, Random Forest, SMOTE, SHAP
- React, Vite
- GitHub Actions

## Notes

- Set `VITE_API_BASE` if the frontend points to a deployed backend.
- Add a real screenshot/GIF and Loom link before publishing.
