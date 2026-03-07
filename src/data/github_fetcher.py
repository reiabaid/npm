"""
GitHub Repository Fetcher
Fetches repository data from the GitHub API.
"""

import requests
import json
import os
from typing import Optional

GITHUB_API_URL = "https://api.github.com"


def fetch_repo_info(owner: str, repo: str, token: Optional[str] = None) -> Optional[dict]:
    """Fetch repository information from GitHub.

    Args:
        owner: The repository owner (user or organization).
        repo: The repository name.
        token: Optional GitHub personal access token for authenticated requests.

    Returns:
        A dictionary containing repository info, or None on failure.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching repo '{owner}/{repo}': {e}")
        return None


def fetch_repo_contributors(owner: str, repo: str, token: Optional[str] = None) -> Optional[list]:
    """Fetch contributors for a GitHub repository.

    Args:
        owner: The repository owner.
        repo: The repository name.
        token: Optional GitHub personal access token.

    Returns:
        A list of contributors, or None on failure.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching contributors for '{owner}/{repo}': {e}")
        return None


def fetch_repo_commits(owner: str, repo: str, token: Optional[str] = None, per_page: int = 30) -> Optional[list]:
    """Fetch recent commits for a GitHub repository.

    Args:
        owner: The repository owner.
        repo: The repository name.
        token: Optional GitHub personal access token.
        per_page: Number of commits to fetch per page.

    Returns:
        A list of commits, or None on failure.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits"
    headers = {"Accept": "application/vnd.github.v3+json"}
    params = {"per_page": per_page}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching commits for '{owner}/{repo}': {e}")
        return None


def save_raw_data(data, filename: str, output_dir: str = "data/raw") -> str:
    """Save raw fetched data to a JSON file.

    Args:
        data: The data to save (dict or list).
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
