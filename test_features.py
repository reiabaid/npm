"""
Quick smoke test: fetch lodash + express from npm & GitHub,
run engineer_features(), and print the full feature dict.
"""

import json
from src.data.npm_fetcher import fetch_npm_raw
from src.data.github_fetcher import fetch_github_stats
from src.data.feature_engineer import engineer_features


def test_package(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")

    # 1. Fetch raw npm JSON
    npm_raw = fetch_npm_raw(name)
    if npm_raw is None:
        print(f"  [SKIP] Could not fetch npm data for '{name}'")
        return

    # 2. Fetch GitHub stats (handles missing repo gracefully)
    repo_field = npm_raw.get("repository")
    github_raw = fetch_github_stats(repo_field)

    # 3. Engineer features
    features = engineer_features(npm_raw, github_raw)

    # 4. Pretty-print the result
    print(json.dumps(features, indent=2, default=str))


if __name__ == "__main__":
    for pkg in ["lodash", "express"]:
        test_package(pkg)
