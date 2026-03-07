"""
NPM Registry Fetcher
Fetches package metadata from the NPM registry API.
"""

import requests
import json
import os
from typing import Optional

NPM_REGISTRY_URL = "https://registry.npmjs.org"


def fetch_package_metadata(package_name: str) -> Optional[dict]:
    """Fetch metadata for a given NPM package.

    Args:
        package_name: The name of the NPM package.

    Returns:
        A dictionary containing the package metadata, or None if not found.
    """
    url = f"{NPM_REGISTRY_URL}/{package_name}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching package '{package_name}': {e}")
        return None


def fetch_package_downloads(package_name: str, period: str = "last-month") -> Optional[dict]:
    """Fetch download statistics for a given NPM package.

    Args:
        package_name: The name of the NPM package.
        period: The time period for download stats (e.g., 'last-month', 'last-week').

    Returns:
        A dictionary containing download statistics, or None on failure.
    """
    url = f"https://api.npmjs.org/downloads/point/{period}/{package_name}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching downloads for '{package_name}': {e}")
        return None


def save_raw_data(data: dict, filename: str, output_dir: str = "data/raw") -> str:
    """Save raw fetched data to a JSON file.

    Args:
        data: The data dictionary to save.
        filename: The output filename.
        output_dir: The directory to save to.

    Returns:
        The path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filepath
