"""NPM Registry Fetcher — fetch, parse, and save npm package metadata."""

import requests
import json
import os
from typing import Optional

NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/point"


def fetch_npm_raw(package_name: str) -> Optional[dict]:
    """Fetch raw metadata for a package from the NPM registry."""
    url = f"{NPM_REGISTRY_URL}/{package_name}"
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
                            period: str = "last-month") -> Optional[dict]:
    """Fetch download statistics for a given NPM package."""
    url = f"{NPM_DOWNLOADS_URL}/{period}/{package_name}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not fetch downloads for '{package_name}': {e}")
        return None


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


def fetch_and_save(package_name: str) -> Optional[dict]:
    """Fetch raw JSON, save it, and return parsed metadata."""
    raw = fetch_npm_raw(package_name)
    if raw is None:
        return None

    save_raw_json(raw, package_name)
    parsed = parse_package_metadata(raw)
    return parsed
