"""GitHub Repository Fetcher — parse npm repo URLs, fetch GitHub stats."""

import re
import os
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_URL = "https://api.github.com"


def parse_github_url(repo_field) -> Optional[Tuple[str, str]]:
    """Extract (owner, repo) from an npm 'repository' field. Returns None if not a GitHub URL."""
    if repo_field is None:
        return None

    if isinstance(repo_field, dict):
        url = repo_field.get("url", "")
    elif isinstance(repo_field, str):
        url = repo_field
    else:
        return None

    if not url:
        return None

    # Strip protocol prefixes and trailing .git
    url = re.sub(r"^git\+", "", url)
    url = re.sub(r"^ssh://git@", "https://", url)
    url = re.sub(r"^git://", "https://", url)
    url = re.sub(r"\.git$", "", url)

    # GitHub shorthand: "github:user/repo"
    shorthand = re.match(r"^github:(.+)/(.+)$", url)
    if shorthand:
        return shorthand.group(1), shorthand.group(2)

    # Full URL: https://github.com/user/repo
    match = re.match(
        r"(?:https?://)?(?:www\.)?github\.com[/:]([^/]+)/([^/#?]+)", url
    )
    if match:
        return match.group(1), match.group(2)

    return None


def fetch_github_raw(owner: str, repo: str) -> Optional[dict]:
    """Fetch raw repo metadata from GET /repos/{owner}/{repo}."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"[ERROR] GitHub repo '{owner}/{repo}' not found (404).")
        else:
            print(f"[ERROR] HTTP error for '{owner}/{repo}': {e}")
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection failed — is your network / GitHub API down?")
        return None
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out for '{owner}/{repo}'.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Unexpected request error for '{owner}/{repo}': {e}")
        return None


def extract_repo_stats(raw: dict) -> dict:
    """Extract stargazers, forks, issues, subscribers, and pushed_at."""
    return {
        "stargazers_count":   raw.get("stargazers_count", 0),
        "forks_count":        raw.get("forks_count", 0),
        "open_issues_count":  raw.get("open_issues_count", 0),
        "subscribers_count":  raw.get("subscribers_count", 0),
        "pushed_at":          raw.get("pushed_at"),
    }


def fetch_contributor_count(owner: str, repo: str) -> int:
    """Get total contributor count using the Link-header pagination trick."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contributors"

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    params = {"per_page": 1, "anon": "true"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        link_header = response.headers.get("Link", "")
        if link_header:
            match = re.search(r'[&?]page=(\d+)[^>]*>;\s*rel="last"', link_header)
            if match:
                return int(match.group(1))

        body = response.json()
        if isinstance(body, list):
            return len(body)
        return 0

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not fetch contributors for '{owner}/{repo}': {e}")
        return 0


def empty_github_stats() -> dict:
    """Return a zeroed-out stats dict for packages with no GitHub repo."""
    return {
        "has_github_repo":    0,
        "stargazers_count":   0,
        "forks_count":        0,
        "open_issues_count":  0,
        "subscribers_count":  0,
        "pushed_at":          None,
        "contributor_count":  0,
    }


def fetch_github_stats(repo_field) -> dict:
    """End-to-end: npm repo field → GitHub stats dict."""
    parsed = parse_github_url(repo_field)
    if parsed is None:
        print("[INFO] No GitHub repo found — returning zeros.")
        return empty_github_stats()

    owner, repo = parsed

    raw = fetch_github_raw(owner, repo)
    if raw is None:
        return empty_github_stats()

    stats = extract_repo_stats(raw)
    stats["contributor_count"] = fetch_contributor_count(owner, repo)
    stats["has_github_repo"] = 1

    return stats
