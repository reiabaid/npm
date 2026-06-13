import requests

def query_osv(package_name, version=None):
    payload = {
        "package" : {"name": package_name, "ecosystem": "npm"}
    }
    if version:
        payload["version"] = version

    resp = requests.post("https://api.osv.dev/v1/query",
                         json=payload, timeout=10)

    return resp.json().get("vulns", [])