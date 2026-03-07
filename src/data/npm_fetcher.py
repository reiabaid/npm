"""
NPM Registry Fetcher
Fetches package metadata from the NPM registry API,
parses key fields, and saves raw JSON locally.
"""

import requests          # Used to make HTTP GET calls to the npm API
import json              # Used to read/write JSON files on disk
import os                # Used to create directories and build file paths
from typing import Optional  # Lets us type-hint that a function may return None


# Base URLs for the two npm APIs we talk to:
# - NPM_REGISTRY_URL  -> gives us all metadata about a package (versions, maintainers, etc.)
# - NPM_DOWNLOADS_URL -> gives us how many times a package was downloaded
NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/point"


# ---------------------------------------------------------------------------
# 1. Fetch raw JSON from the NPM registry
# ---------------------------------------------------------------------------

# This function takes a package name (like "express"), calls the npm registry
# API, and returns the entire JSON response as a Python dictionary.
# If anything goes wrong (package doesn't exist, network is down, timeout),
# it prints a helpful error message and returns None instead of crashing.
def fetch_npm_raw(package_name: str) -> Optional[dict]:
    """Fetch raw metadata for a package from the NPM registry.

    Hits https://registry.npmjs.org/{package_name} and returns the
    full JSON response as a dictionary.

    Args:
        package_name: The name of the NPM package (e.g. "express").

    Returns:
        The raw JSON response as a dict, or None if the request fails.
    """
    # Build the full URL, e.g. https://registry.npmjs.org/express
    url = f"{NPM_REGISTRY_URL}/{package_name}"
    try:
        # Send the GET request; give up if it takes longer than 30 seconds
        response = requests.get(url, timeout=30)
        # If the server returned an error status (4xx/5xx), this will raise an exception
        response.raise_for_status()
        # Convert the JSON string in the response body into a Python dict and return it
        return response.json()
    except requests.exceptions.HTTPError as e:
        # The server responded but with an error code
        if e.response is not None and e.response.status_code == 404:
            # 404 means the package name doesn't exist on npm
            print(f"[ERROR] Package '{package_name}' does not exist on npm.")
        else:
            print(f"[ERROR] HTTP error for '{package_name}': {e}")
        return None
    except requests.exceptions.ConnectionError:
        # Could not reach the server at all (no internet, DNS failure, etc.)
        print(f"[ERROR] Connection failed -- is your network/npm API down?")
        return None
    except requests.exceptions.Timeout:
        # The server took longer than 30 seconds to respond
        print(f"[ERROR] Request timed out for '{package_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        # Catch-all for any other request-related error we didn't anticipate
        print(f"[ERROR] Unexpected request error for '{package_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# 2. Parse the raw JSON and extract key fields
# ---------------------------------------------------------------------------

# The raw JSON from npm contains hundreds of fields. This function picks out
# only the ones we care about: name, description, when it was created/modified,
# timestamps for every version ever published, who maintains it, keywords,
# and the build/test scripts from the latest version.
def parse_package_metadata(raw: dict) -> dict:
    """Extract key fields from the raw NPM registry JSON.

    Fields extracted:
        - name, description
        - created & modified timestamps
        - all version timestamps (from the 'time' object)
        - maintainers list
        - keywords
        - scripts object from the *latest* published version

    Args:
        raw: The full JSON dict returned by fetch_npm_raw().

    Returns:
        A dict with the extracted fields.
    """
    # The "time" object maps version numbers to their publish timestamps,
    # plus two special keys: "created" (first publish) and "modified" (last update)
    time_obj = raw.get("time", {})

    # Build a dict of ONLY version->timestamp pairs (skip "created"/"modified")
    version_timestamps = {
        ver: ts for ver, ts in time_obj.items()
        if ver not in ("created", "modified")
    }

    # Find the latest version number (e.g. "4.17.23") from dist-tags,
    # then look up that version's full data to grab its scripts
    latest_tag = raw.get("dist-tags", {}).get("latest", "")
    versions = raw.get("versions", {})
    latest_version_data = versions.get(latest_tag, {})
    scripts = latest_version_data.get("scripts", {})

    # Return a clean, flat dictionary with just the fields we need
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


# ---------------------------------------------------------------------------
# 3. Fetch download statistics
# ---------------------------------------------------------------------------

# This function calls a DIFFERENT npm API to get download counts.
# For example, "how many times was axios downloaded in the last month?"
# Defaults to "last-month" but you can also pass "last-week" or "last-year".
def fetch_package_downloads(package_name: str,
                            period: str = "last-month") -> Optional[dict]:
    """Fetch download statistics for a given NPM package.

    Args:
        package_name: The name of the NPM package.
        period: Time period (e.g. 'last-month', 'last-week').

    Returns:
        A dict with download stats, or None on failure.
    """
    url = f"{NPM_DOWNLOADS_URL}/{period}/{package_name}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not fetch downloads for '{package_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Save raw JSON to data/raw/{package_name}.json
# ---------------------------------------------------------------------------

# This function takes any dictionary and writes it to a .json file on disk.
# It creates the output folder if it doesn't exist yet.
# For scoped packages like "@angular/core", it replaces special characters
# so the filename is valid (e.g. angular_core.json).
def save_raw_json(data: dict, package_name: str,
                  output_dir: str = "data/raw") -> str:
    """Save raw fetched data to data/raw/{package_name}.json.

    Args:
        data: The data dictionary to save.
        package_name: Used to build the filename.
        output_dir: The directory to save to (created if missing).

    Returns:
        The path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Handle scoped packages like @angular/core -> angular_core.json
    safe_name = package_name.replace("/", "_").replace("@", "")
    filepath = os.path.join(output_dir, f"{safe_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVED] {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# 5. High-level pipeline: fetch + parse + save
# ---------------------------------------------------------------------------

# This is the main "do everything" function. It:
#   1) Fetches the raw JSON from npm
#   2) Saves that raw JSON to data/raw/ as a backup
#   3) Parses out the important fields and returns them
# If the fetch fails, it returns None so the caller can skip that package.
def fetch_and_save(package_name: str) -> Optional[dict]:
    """Fetch a package's raw JSON, save it, and return parsed metadata.

    Args:
        package_name: NPM package name.

    Returns:
        Parsed metadata dict, or None if the fetch failed.
    """
    # Step 1: Hit the npm API and get the full JSON
    raw = fetch_npm_raw(package_name)
    if raw is None:
        # fetch failed (error already printed), so bail out
        return None

    # Step 2: Write the raw JSON to disk for reference
    save_raw_json(raw, package_name)
    # Step 3: Pull out the fields we need and return them
    parsed = parse_package_metadata(raw)
    return parsed
