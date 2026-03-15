import os
import time
import pandas as pd
from typing import List

from src.data.npm_fetcher import fetch_npm_raw, save_raw_json
from src.data.github_fetcher import fetch_github_stats
from src.data.feature_engineer import engineer_features

def load_packages(filepath: str) -> List[str]:
    """Helper to read packages from text file, ignoring empty lines & comments."""
    if not os.path.exists(filepath):
        print(f"[WARNING] {filepath} does not exist.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def main():
    healthy_path = os.path.join("data", "healthy_packages.txt")
    suspicious_path = os.path.join("data", "suspicious_packages.txt")
    failed_log = "failed_packages.txt"
    output_path = os.path.join("data", "processed", "dataset.csv")

    healthy_pkgs = load_packages(healthy_path)
    suspicious_pkgs = load_packages(suspicious_path)

    # Label: 0 for healthy, 1 for suspicious
    all_pkgs = [(pkg, 0) for pkg in healthy_pkgs] + [(pkg, 1) for pkg in suspicious_pkgs]

    processed_data = []

    print(f"Total packages to process: {len(all_pkgs)}")

    for i, (pkg, label) in enumerate(all_pkgs):
        print(f"[{i+1}/{len(all_pkgs)}] Processing {pkg} (label {label})...")
        try:
            # 1. NPM fetch
            npm_raw = fetch_npm_raw(pkg)
            if npm_raw is None:
                raise ValueError(f"Failed to fetch NPM metadata for {pkg}")
            
            # Save raw jsons
            save_raw_json(npm_raw, pkg)
            time.sleep(0.5)

            # 2. GitHub fetch
            repository_info = npm_raw.get('repository')
            github_stats = fetch_github_stats(repository_info)
            time.sleep(1.0)
            
            # 3. Engineer features
            features = engineer_features(npm_raw, github_stats)
            
            # 4. Add label
            features['label'] = label
            
            processed_data.append(features)

        except Exception as e:
            print(f"[ERROR] processing {pkg}: {e}")
            with open(failed_log, 'a', encoding='utf-8') as f:
                f.write(f"{pkg}\n")

    # 5. Save to CSV
    if processed_data:
        df = pd.DataFrame(processed_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved dataset with {len(processed_data)} rows to {output_path}")
    else:
        print("No packages were successfully processed.")

if __name__ == "__main__":
    main()
