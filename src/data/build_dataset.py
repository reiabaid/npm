import os
import time
import pandas as pd
from typing import List, Tuple

from src.data.npm_fetcher import (
    fetch_npm_raw,
    fetch_package_downloads,
    fetch_maintainer_age,
    save_raw_json,
)
from src.data.github_fetcher import fetch_github_stats
from src.data.feature_engineer import engineer_features


def load_packages(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print(f"[WARNING] {filepath} does not exist.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_confirmed_malicious(filepath: str) -> List[str]:
    """Load only the confirmed-malicious section of suspicious_packages.txt.

    The file has two clearly labelled sections:
        # -- Confirmed Malicious Packages --
        # -- Synthetic Typosquats (generated from top 50) --

    We stop reading at the Synthetic Typosquats header so fabricated names
    never enter the training set.
    """
    packages = []
    in_confirmed = False
    if not os.path.exists(filepath):
        print(f"[WARNING] {filepath} does not exist.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Confirmed Malicious" in line:
                in_confirmed = True
            elif "Synthetic Typosquats" in line:
                break
            elif in_confirmed and line and not line.startswith("#"):
                packages.append(line)
    return packages


def process_package(pkg: str, label: int, popular_names: List[str]) -> dict | None:
    """Fetch all data for one package and return a feature dict, or None on failure."""
    npm_raw = fetch_npm_raw(pkg)
    if npm_raw is None:
        raise ValueError(f"npm fetch returned None for '{pkg}'")

    save_raw_json(npm_raw, pkg)
    time.sleep(0.3)

    repository_info = npm_raw.get("repository")
    github_stats = fetch_github_stats(repository_info)
    time.sleep(0.5)

    weekly_downloads = fetch_package_downloads(pkg)
    time.sleep(0.3)

    maintainers = npm_raw.get("maintainers", [])
    maintainer_min_age = fetch_maintainer_age(maintainers)
    time.sleep(0.3)

    features = engineer_features(
        npm_raw,
        github_stats,
        weekly_downloads=weekly_downloads,
        maintainer_min_age_days=maintainer_min_age,
        popular_names=popular_names,
    )
    features["label"] = label
    return features


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    healthy_path         = os.path.join(base_dir, "data", "healthy_packages.txt")
    hard_negatives_path  = os.path.join(base_dir, "data", "hard_negatives.txt")
    suspicious_path      = os.path.join(base_dir, "data", "suspicious_packages.txt")
    failed_log           = os.path.join(base_dir, "failed_packages.txt")
    output_path          = os.path.join(base_dir, "data", "processed", "dataset.csv")

    healthy_pkgs    = load_packages(healthy_path)
    hard_neg_pkgs   = load_packages(hard_negatives_path)
    # Only confirmed malicious — no synthetic typosquats in training data
    malicious_pkgs  = load_confirmed_malicious(suspicious_path)

    popular_names = list(healthy_pkgs)  # used for typosquat distance feature

    # label=0: top-500 healthy + hard negatives (small but legitimate)
    # label=1: confirmed malicious packages only
    all_pkgs: List[Tuple[str, int]] = (
        [(p, 0) for p in healthy_pkgs]
        + [(p, 0) for p in hard_neg_pkgs]
        + [(p, 1) for p in malicious_pkgs]
    )

    # Deduplicate by package name (keep first occurrence)
    seen: set = set()
    deduped = []
    for pkg, label in all_pkgs:
        if pkg not in seen:
            seen.add(pkg)
            deduped.append((pkg, label))
    all_pkgs = deduped

    processed_data = []
    print(f"Total packages to process: {len(all_pkgs)}")
    print(f"  Healthy (top-500):   {sum(1 for _, l in all_pkgs if l == 0 and _ in set(healthy_pkgs))}")
    print(f"  Hard negatives:      {sum(1 for p, l in all_pkgs if l == 0 and p in set(hard_neg_pkgs))}")
    print(f"  Confirmed malicious: {sum(1 for _, l in all_pkgs if l == 1)}")

    for i, (pkg, label) in enumerate(all_pkgs):
        print(f"[{i+1}/{len(all_pkgs)}] Processing {pkg} (label={label})...")
        try:
            features = process_package(pkg, label, popular_names)
            processed_data.append(features)
        except Exception as e:
            print(f"[ERROR] {pkg}: {e}")
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{pkg}\n")

    if processed_data:
        df = pd.DataFrame(processed_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(processed_data)} rows to {output_path}")
        print(f"Label counts:\n{df['label'].value_counts().to_string()}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No packages were successfully processed.")


if __name__ == "__main__":
    main()
