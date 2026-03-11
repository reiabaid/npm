# SCOPE - Security Check for Open-source Package Ecosystems

A security analysis tool for detecting malicious NPM packages using machine learning.

## Project Structure

```
scope/
├── data/
│   ├── raw/           # Raw data from NPM and GitHub
│   └── processed/     # Processed and engineered features
├── src/
│   ├── data/
│   │   ├── npm_fetcher.py       # Fetch package metadata from NPM registry
│   │   ├── github_fetcher.py    # Fetch repository data from GitHub API
│   │   └── feature_engineer.py  # Engineer features for model training
│   ├── model/
│   │   ├── train.py             # Train the detection model
│   │   ├── evaluate.py          # Evaluate model performance
│   │   └── explain.py           # Model explainability utilities
│   ├── cli/
│   │   └── scope.py             # Command-line interface
│   └── api/
│       └── main.py              # REST API server
├── models/                      # Saved model artifacts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python -m src.cli.scope <package-name>
```

### API

```bash
python -m src.api.main
```
