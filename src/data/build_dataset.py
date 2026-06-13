import os
import json
import time
import threading
import pandas as pd
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.npm_fetcher import (
    fetch_npm_raw,
    fetch_package_downloads,
    fetch_maintainer_age,
    save_raw_json,
    load_raw_json,
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


def _meta_cache_path(pkg: str, base_dir: str = "data/raw") -> str:
    safe = pkg.replace("/", "_").replace("@", "")
    return os.path.join(base_dir, f"_meta_{safe}.json")


def process_package(pkg: str, label: int, popular_names: List[str]) -> dict | None:
    """Fetch all data for one package and return a feature dict, or None on failure."""
    npm_raw = load_raw_json(pkg) or fetch_npm_raw(pkg)
    if npm_raw is None:
        raise ValueError(f"npm fetch returned None for '{pkg}'")

    if not load_raw_json(pkg):
        save_raw_json(npm_raw, pkg)

    repository_info = npm_raw.get("repository")
    maintainers = npm_raw.get("maintainers", [])

    meta_path = _meta_cache_path(pkg)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        github_stats = cached["github"]
        weekly_downloads = cached["downloads"]
        maintainer_min_age = cached["maintainer_age"]
    else:
        github_stats = fetch_github_stats(repository_info)
        weekly_downloads = fetch_package_downloads(pkg)
        maintainer_min_age = fetch_maintainer_age(maintainers)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"github": github_stats, "downloads": weekly_downloads,
                       "maintainer_age": maintainer_min_age}, f)

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
    healthy_path              = os.path.join(base_dir, "data", "healthy_packages.txt")
    hard_negatives_path       = os.path.join(base_dir, "data", "hard_negatives.txt")
    challenging_negatives_path = os.path.join(base_dir, "data", "challenging_negatives.txt")
    suspicious_path           = os.path.join(base_dir, "data", "suspicious_packages.txt")
    failed_log           = os.path.join(base_dir, "failed_packages.txt")
    output_path          = os.path.join(base_dir, "data", "processed", "dataset.csv")

    healthy_pkgs          = load_packages(healthy_path)
    hard_neg_pkgs         = load_packages(hard_negatives_path)
    challenging_neg_pkgs  = load_packages(challenging_negatives_path)
    # Only confirmed malicious — no synthetic typosquats in training data
    malicious_pkgs        = load_confirmed_malicious(suspicious_path)

    popular_names = list(healthy_pkgs)  # used for typosquat distance feature

    # Malicious packages first so GitHub rate limiting doesn't prevent them from
    # being processed — healthy packages have disk-cached npm JSON already.
    all_pkgs: List[Tuple[str, int]] = (
        [(p, 1) for p in malicious_pkgs]
        + [(p, 0) for p in healthy_pkgs]
        + [(p, 0) for p in hard_neg_pkgs]
        + [(p, 0) for p in challenging_neg_pkgs]
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
    total = len(all_pkgs)
    print(f"Total packages to process: {total}")
    print(f"  Healthy (top-500):      {sum(1 for p, l in all_pkgs if l == 0 and p in set(healthy_pkgs))}")
    print(f"  Hard negatives:         {sum(1 for p, l in all_pkgs if l == 0 and p in set(hard_neg_pkgs))}")
    print(f"  Challenging negatives:  {sum(1 for p, l in all_pkgs if l == 0 and p in set(challenging_neg_pkgs))}")
    print(f"  Confirmed malicious:    {sum(1 for _, l in all_pkgs if l == 1)}")

    lock = threading.Lock()
    completed = 0

    def process_one(args):
        nonlocal completed
        pkg, label = args
        features = process_package(pkg, label, popular_names)
        with lock:
            completed += 1
            print(f"[{completed}/{total}] OK {pkg} (label={label})", flush=True)
        return features

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one, (pkg, label)): pkg for pkg, label in all_pkgs}
        for future in as_completed(futures):
            pkg = futures[future]
            try:
                processed_data.append(future.result())
            except Exception as e:
                print(f"[ERROR] {pkg}: {e}", flush=True)
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
