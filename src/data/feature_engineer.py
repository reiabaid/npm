"""Feature Engineering — compute one flat feature dict from raw NPM + GitHub data."""

from datetime import datetime, timezone

STANDARD_LICENSES = {"MIT", "ISC", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}


def _parse_iso(date_str: str) -> datetime:
    """Parse an ISO-8601 date string (with trailing Z) into a UTC datetime."""
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def _days_since(date_str) -> int:
    """Return whole days between date_str and now. Returns 0 if None/invalid."""
    if not date_str:
        return 0
    try:
        return max((datetime.now(timezone.utc) - _parse_iso(date_str)).days, 0)
    except (ValueError, TypeError):
        return 0


def engineer_features(npm_raw: dict, github_raw: dict) -> dict:
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
    has_postinstall = 1 if "postinstall" in scripts else 0
    description_length = len(npm_raw.get("description", "") or "")

    raw_license = npm_raw.get("license", "")
    if isinstance(raw_license, dict):
        raw_license = raw_license.get("type", "")
    license_is_standard = 1 if raw_license in STANDARD_LICENSES else 0

    days_since_last_commit = _days_since(github_raw.get("pushed_at"))

    return {
        "name":                   npm_raw.get("name", ""),
        "days_since_created":     days_since_created,
        "days_since_last_update": days_since_last_update,
        "num_versions":           num_versions,
        "release_velocity":       round(release_velocity, 6),
        "num_maintainers":        num_maintainers,
        "has_postinstall":        has_postinstall,
        "description_length":     description_length,
        "license_is_standard":    license_is_standard,
        "has_github_repo":        github_raw.get("has_github_repo", 0),
        "stargazers_count":       github_raw.get("stargazers_count", 0),
        "forks_count":            github_raw.get("forks_count", 0),
        "open_issues_count":      github_raw.get("open_issues_count", 0),
        "subscribers_count":      github_raw.get("subscribers_count", 0),
        "contributor_count":      github_raw.get("contributor_count", 0),
        "days_since_last_commit": days_since_last_commit,
    }
