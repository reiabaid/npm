"""
Feature Engineering
Transforms raw NPM and GitHub data into features for model training.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional


def extract_npm_features(metadata: dict) -> dict:
    """Extract features from NPM package metadata.

    Args:
        metadata: Raw NPM package metadata dictionary.

    Returns:
        A dictionary of extracted features.
    """
    latest_version = metadata.get("dist-tags", {}).get("latest", "")
    versions = metadata.get("versions", {})
    maintainers = metadata.get("maintainers", [])
    time_info = metadata.get("time", {})

    features = {
        "name": metadata.get("name", ""),
        "version_count": len(versions),
        "maintainer_count": len(maintainers),
        "has_readme": bool(metadata.get("readme")),
        "has_homepage": bool(metadata.get("homepage")),
        "has_repository": bool(metadata.get("repository")),
        "has_bugs_url": bool(metadata.get("bugs")),
        "has_license": bool(metadata.get("license")),
        "latest_version": latest_version,
    }

    # Dependency analysis for the latest version
    if latest_version and latest_version in versions:
        latest = versions[latest_version]
        features["dependency_count"] = len(latest.get("dependencies", {}))
        features["dev_dependency_count"] = len(latest.get("devDependencies", {}))
        features["has_scripts"] = bool(latest.get("scripts"))
        features["has_install_script"] = "install" in latest.get("scripts", {}) or \
                                          "preinstall" in latest.get("scripts", {}) or \
                                          "postinstall" in latest.get("scripts", {})

    return features


def extract_github_features(repo_info: dict, contributors: Optional[list] = None) -> dict:
    """Extract features from GitHub repository data.

    Args:
        repo_info: Raw GitHub repository info dictionary.
        contributors: Optional list of contributors.

    Returns:
        A dictionary of extracted features.
    """
    features = {
        "stars": repo_info.get("stargazers_count", 0),
        "forks": repo_info.get("forks_count", 0),
        "open_issues": repo_info.get("open_issues_count", 0),
        "watchers": repo_info.get("watchers_count", 0),
        "is_fork": repo_info.get("fork", False),
        "is_archived": repo_info.get("archived", False),
        "has_wiki": repo_info.get("has_wiki", False),
        "has_pages": repo_info.get("has_pages", False),
        "repo_size": repo_info.get("size", 0),
        "default_branch": repo_info.get("default_branch", ""),
    }

    if contributors:
        features["contributor_count"] = len(contributors)

    return features


def build_feature_dataframe(records: list[dict]) -> pd.DataFrame:
    """Build a pandas DataFrame from a list of feature dictionaries.

    Args:
        records: A list of feature dictionaries.

    Returns:
        A pandas DataFrame of features.
    """
    df = pd.DataFrame(records)
    return df


def save_processed_data(df: pd.DataFrame, filename: str, output_dir: str = "data/processed") -> str:
    """Save processed features to a CSV file.

    Args:
        df: The DataFrame to save.
        filename: The output filename.
        output_dir: The directory to save to.

    Returns:
        The path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath
