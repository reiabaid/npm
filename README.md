# Sentinel NPM

A security analysis tool for NPM packages.

## Project Structure

```
sentinel-npm/
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
│   │   └── sentinel.py          # Command-line interface
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
python -m src.cli.sentinel <package-name>
```

### API

```bash
python -m src.api.main
```
