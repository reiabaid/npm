import os
import time
import pandas as pd
from typing import List

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


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    healthy_path    = os.path.join(base_dir, "data", "healthy_packages.txt")
    suspicious_path = os.path.join(base_dir, "data", "suspicious_packages.txt")
    failed_log      = os.path.join(base_dir, "failed_packages.txt")
    output_path     = os.path.join(base_dir, "data", "processed", "dataset.csv")

    healthy_pkgs    = load_packages(healthy_path)
    suspicious_pkgs = load_packages(suspicious_path)

    # Load popular package names for typosquat distance computation
    popular_names = [p for p in healthy_pkgs]

    all_pkgs = [(pkg, 0) for pkg in healthy_pkgs] + [(pkg, 1) for pkg in suspicious_pkgs]
    processed_data = []

    print(f"Total packages to process: {len(all_pkgs)}")

    for i, (pkg, label) in enumerate(all_pkgs):
        print(f"[{i+1}/{len(all_pkgs)}] Processing {pkg} (label={label})...")
        try:
            npm_raw = fetch_npm_raw(pkg)
            if npm_raw is None:
                raise ValueError(f"Failed to fetch npm metadata for {pkg}")

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
            processed_data.append(features)

        except Exception as e:
            print(f"[ERROR] processing {pkg}: {e}")
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{pkg}\n")

    if processed_data:
        df = pd.DataFrame(processed_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved dataset with {len(processed_data)} rows to {output_path}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No packages were successfully processed.")


if __name__ == "__main__":
    main()
