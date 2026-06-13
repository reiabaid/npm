"""NPM Registry Fetcher — fetch, parse, and save npm package metadata."""

import requests
import json
import os
from typing import Optional

NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/point"


def fetch_npm_raw(package_name: str) -> Optional[dict]:
    """Fetch raw metadata for a package from the NPM registry."""
    if not package_name or not package_name.strip():
        return None
    url = f"{NPM_REGISTRY_URL}/{package_name.strip()}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"[ERROR] Package '{package_name}' does not exist on npm.")
        else:
            print(f"[ERROR] HTTP error for '{package_name}': {e}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Connection failed -- is your network/npm API down?")
        return None
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out for '{package_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Unexpected request error for '{package_name}': {e}")
        return None


def parse_package_metadata(raw: dict) -> dict:
    """Extract key fields (name, timestamps, maintainers, scripts, etc.)."""
    time_obj = raw.get("time", {})

    version_timestamps = {
        ver: ts for ver, ts in time_obj.items()
        if ver not in ("created", "modified")
    }

    latest_tag = raw.get("dist-tags", {}).get("latest", "")
    latest_version_data = raw.get("versions", {}).get(latest_tag, {})
    scripts = latest_version_data.get("scripts", {})

    return {
        "name":                raw.get("name"),
        "description":         raw.get("description"),
        "created":             time_obj.get("created"),
        "modified":            time_obj.get("modified"),
        "version_timestamps":  version_timestamps,
        "maintainers":         raw.get("maintainers", []),
        "keywords":            raw.get("keywords", []),
        "latest_version":      latest_tag,
        "scripts":             scripts,
    }


def fetch_package_downloads(package_name: str,
                            period: str = "last-week") -> int:
    """Return weekly download count for a package, or 0 on failure."""
    url = f"{NPM_DOWNLOADS_URL}/{period}/{package_name}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json().get("downloads", 0) or 0
    except requests.exceptions.RequestException:
        return 0


def fetch_maintainer_age(maintainers: list) -> int:
    """Return the age in days of the *newest* maintainer account (min across all).

    Uses the npm user profile endpoint. Returns 0 if no accounts can be fetched,
    which is treated as maximally suspicious (unknown/new account).
    """
    if not maintainers:
        return 0

    from datetime import datetime, timezone

    min_age = None
    for m in maintainers:
        username = m.get("name") if isinstance(m, dict) else str(m)
        if not username:
            continue
        url = f"{NPM_REGISTRY_URL}/-/user/org.couchdb.user/{username}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                created_str = resp.json().get("created")
                if created_str:
                    created = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                    age = (datetime.now(timezone.utc) - created).days
                    if min_age is None or age < min_age:
                        min_age = age
        except requests.exceptions.RequestException:
            continue

    return min_age if min_age is not None else 0


def save_raw_json(data: dict, package_name: str,
                  output_dir: str = "data/raw") -> str:
    """Save raw fetched data to data/raw/{package_name}.json."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = package_name.replace("/", "_").replace("@", "")
    filepath = os.path.join(output_dir, f"{safe_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVED] {filepath}")
    return filepath


def load_raw_json(package_name: str, output_dir: str = "data/raw") -> Optional[dict]:
    """Load previously saved raw JSON from disk, or return None if not cached."""
    safe_name = package_name.replace("/", "_").replace("@", "")
    filepath = os.path.join(output_dir, f"{safe_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_and_save(package_name: str) -> Optional[dict]:
    """Fetch raw JSON, save it, and return parsed metadata."""
    raw = fetch_npm_raw(package_name)
    if raw is None:
        return None

    save_raw_json(raw, package_name)
    parsed = parse_package_metadata(raw)
    return parsed
