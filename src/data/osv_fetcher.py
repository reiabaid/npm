import requests

_cache: dict = {}

def query_osv(package_name, version=None):
    key = f"{package_name}@{version or '*'}"
    if key in _cache:
        return _cache[key]

    payload = {"package": {"name": package_name, "ecosystem": "npm"}}
    if version:
        payload["version"] = version

    resp = requests.post("https://api.osv.dev/v1/query",
                         json=payload, timeout=10)
    result = resp.json().get("vulns", [])
    _cache[key] = result
    return result
