"""Feature Engineering — compute one flat feature dict from raw NPM + GitHub data."""

import difflib
from datetime import datetime, timezone

STANDARD_LICENSES = {"MIT", "ISC", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}

DANGER_PATTERNS = [
    "curl ", "wget ", "fetch(", "base64",
    "/tmp/", "process.env", "child_process",
    "exec(", "spawn(", "eval(",
]

_INSTALL_HOOKS = ("preinstall", "install", "postinstall")


def _parse_iso(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def _days_since(date_str) -> int:
    if not date_str:
        return 0
    try:
        return max((datetime.now(timezone.utc) - _parse_iso(date_str)).days, 0)
    except (ValueError, TypeError):
        return 0


def _min_edit_distance(name: str, popular_names: list) -> int:
    """Character-level edit distance from `name` to its closest popular package.

    Uses SequenceMatcher ratio → estimated edit distance so no external dep is
    needed.  Returns len(name) when no popular names are provided (neutral).
    """
    if not popular_names:
        return len(name)
    best_ratio = 0.0
    for pop in popular_names:
        r = difflib.SequenceMatcher(None, name, pop).ratio()
        if r > best_ratio:
            best_ratio = r
    avg_len = (len(name) + 1) / 2  # rough denominator
    return max(0, round(avg_len * (1 - best_ratio)))


def _script_suspicion_score(scripts: dict) -> int:
    """Count how many danger patterns appear across all lifecycle script values."""
    combined = " ".join(
        v for k, v in scripts.items() if isinstance(v, str)
    ).lower()
    return sum(1 for pat in DANGER_PATTERNS if pat in combined)


def engineer_features(
    npm_raw: dict,
    github_raw: dict,
    weekly_downloads: int = 0,
    maintainer_min_age_days: int = 0,
    popular_names: list = None,
) -> dict:
    """Compute one clean feature dict from raw NPM + GitHub data."""
    time_obj = npm_raw.get("time", {})
    versions = npm_raw.get("versions", {})
    maintainers = npm_raw.get("maintainers", [])

    latest_tag = npm_raw.get("dist-tags", {}).get("latest", "")
    latest_version_data = versions.get(latest_tag, {})
    scripts = latest_version_data.get("scripts", {})

    days_since_created = _days_since(time_obj.get("created"))
    days_since_last_update = _days_since(time_obj.get(latest_tag))
    num_versions = len(versions)
    release_velocity = num_versions / max(days_since_created, 1)
    num_maintainers = len(maintainers)
    description_length = len(npm_raw.get("description", "") or "")

    raw_license = npm_raw.get("license", "")
    if isinstance(raw_license, dict):
        raw_license = raw_license.get("type", "")
    license_is_standard = 1 if raw_license in STANDARD_LICENSES else 0

    days_since_last_commit = _days_since(github_raw.get("pushed_at"))

    pkg_name = npm_raw.get("name", "")
    typosquat_min_distance = _min_edit_distance(pkg_name, popular_names or [])

    has_any_install_hook = 1 if any(h in scripts for h in _INSTALL_HOOKS) else 0
    script_suspicion_score = _script_suspicion_score(scripts)

    return {
        "days_since_created":           days_since_created,
        "days_since_last_update":       days_since_last_update,
        "num_versions":                 num_versions,
        "release_velocity":             round(release_velocity, 6),
        "num_maintainers":              num_maintainers,
        "description_length":           description_length,
        "weekly_downloads":             int(weekly_downloads),
        "typosquat_min_distance":       typosquat_min_distance,
        "script_suspicion_score":       script_suspicion_score,
        "maintainer_min_account_age_days": int(maintainer_min_age_days),
        "stargazers_count":             github_raw.get("stargazers_count", 0),
        "forks_count":                  github_raw.get("forks_count", 0),
        "open_issues_count":            github_raw.get("open_issues_count", 0),
        "subscribers_count":            github_raw.get("subscribers_count", 0),
        "contributor_count":            github_raw.get("contributor_count", 0),
        "days_since_last_commit":       days_since_last_commit,
        "has_any_install_hook":         has_any_install_hook,
        "license_is_standard":          license_is_standard,
        "has_github_repo":              github_raw.get("has_github_repo", 0),
    }
