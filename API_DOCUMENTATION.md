# SCOPE API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Interactive API Docs:** `http://localhost:8000/docs` (Swagger UI)

## Overview

SCOPE API is a REST interface for AI-powered NPM package security analysis. The API analyzes packages for malicious indicators using machine learning and provides risk scores with SHAP-based feature explanations.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Endpoints](#endpoints)
3. [Request/Response Models](#requestresponse-models)
4. [Error Handling](#error-handling)
5. [Features](#features)
6. [Examples](#examples)
7. [Rate Limiting & Caching](#rate-limiting--caching)

---

## Getting Started

### Starting the API

```bash
# Start development server with hot reload
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Start production server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API status and model availability

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2024-04-12T19:21:20.030039"
}
```

**Status Codes:** `200 OK`

---

### 2. Analyze Single Package

**Endpoint:** `POST /analyze`

**Description:** Analyze a single NPM package for security risks

**Request Body:**
```json
{
  "package_name": "express"
}
```

**Response:** `AnalysisResult`
```json
{
  "package": "express",
  "score": 0.0011,
  "risk_level": "HEALTHY",
  "features": {
    "name": "express",
    "days_since_created": 5582,
    "days_since_last_update": 131,
    "num_versions": 287,
    "release_velocity": 0.051415,
    "num_maintainers": 5,
    "has_postinstall": 0,
    "description_length": 45,
    "license_is_standard": 1,
    "has_github_repo": 1,
    "stargazers_count": 68924,
    "forks_count": 23088,
    "open_issues_count": 208,
    "subscribers_count": 1674,
    "contributor_count": 380,
    "days_since_last_commit": 5
  },
  "explanations": [
    {
      "feature": "num_versions",
      "shap_value": -1.647,
      "description": null
    }
  ],
  "warnings": [],
  "suggestion": null,
  "error": null,
  "status": null
}
```

**Status Codes:**
- `200 OK` — Analysis successful
- `404 Not Found` — Package not found on npm
- `422 Unprocessable Entity` — Invalid request (missing/empty package name)
- `503 Service Unavailable` — Model not initialized
- `500 Internal Server Error` — Server error during analysis

**Caching:** Results are cached for 30 minutes. Subsequent requests for the same package return cached data.

---

### 3. Analyze Multiple Packages (Batch)

**Endpoint:** `POST /batch`

**Description:** Analyze multiple packages in a single request (max 20)

**Request Body:**
```json
{
  "packages": ["express", "react", "vue", "lodash"]
}
```

**Response:** `List[AnalysisResult]`
```json
[
  {
    "package": "express",
    "score": 0.0011,
    "risk_level": "HEALTHY",
    ...
  },
  {
    "package": "react",
    "score": 0.0013,
    "risk_level": "HEALTHY",
    ...
  }
]
```

**Constraints:**
- Minimum 1 package
- Maximum 20 packages
- Duplicates are automatically removed

**Status Codes:**
- `200 OK` — Analysis successful (even if some packages fail)
- `422 Unprocessable Entity` — Invalid request (too many packages, empty list, invalid format)
- `503 Service Unavailable` — Model not initialized
- `500 Internal Server Error` — Server error during analysis

**Note:** Failed packages are included in the response with error information.

---

### 4. Clear Cache

**Endpoint:** `GET /cache/clear`

**Description:** Clear all cached analysis results

**Response:**
```json
{
  "status": "ok",
  "message": "Cache cleared"
}
```

**Status Codes:** `200 OK`

---

### 5. Get Cache Size

**Endpoint:** `GET /cache/size`

**Description:** Get current cache statistics

**Response:**
```json
{
  "cached_packages": 5,
  "cache_ttl_seconds": 1800
}
```

**Status Codes:** `200 OK`

---

## Request/Response Models

### PackageRequest
```python
{
  "package_name": str  # Package name (required, non-empty)
}
```

### BatchRequest
```python
{
  "packages": List[str]  # 1-20 package names (required)
}
```

### AnalysisResult
```python
{
  "package": str                    # Package name
  "score": float                    # Risk score (0.0-1.0)
  "risk_level": str                 # HEALTHY | MEDIUM | HIGH | CRITICAL
  "features": Optional[Dict]        # Analyzed features
  "explanations": List[ShapFactor]  # Top 4 feature impacts
  "warnings": List[str]             # Any warnings during analysis
  "suggestion": Optional[Dict]      # Typosquatting suggestions (if CRITICAL)
  "error": Optional[str]            # Error message (if failed)
  "status": Optional[str]           # Error status code (NOT_FOUND | ERROR)
}
```

### ShapFactor
```python
{
  "feature": str              # Feature name
  "shap_value": float         # SHAP contribution value
  "description": Optional[str] # Feature description (null)
}
```

### HealthResponse
```python
{
  "status": str      # "ok"
  "model_loaded": bool # True if model is loaded
  "timestamp": str   # ISO 8601 timestamp
}
```

---

## Error Handling

### Error Response Format

All errors return a JSON object with details:

```json
{
  "detail": "Error description or array of validation errors"
}
```

### Common Errors

| Status | Scenario | Detail |
|--------|----------|--------|
| 404 | Package not found | `Package 'xyz' not found on npm` |
| 422 | Invalid input | `List should have at most 20 items` |
| 503 | Model not ready | `SCOPE engine not initialized` |
| 500 | Server error | `Internal server error: {error details}` |

---

## Features

### Risk Scoring

The API returns risk scores from `0.0` (healthy) to `1.0` (critical):

| Score Range | Risk Level | Interpretation |
|-------------|-----------|-----------------|
| 0.0 - 0.2  | HEALTHY   | No detectable risk |
| 0.2 - 0.5  | MEDIUM    | Minor risk indicators |
| 0.5 - 0.8  | HIGH      | Significant risk factors |
| 0.8 - 1.0  | CRITICAL  | Likely malicious |

### Feature Analysis

The analysis examines 15 package features:

- **Temporal:** Days since creation, last update, release velocity
- **Community:** Number of maintainers, contributors, GitHub stars
- **Repository:** Forks, issues, subscribers, last commit
- **Package:** Versions, description length, postinstall scripts
- **Licensing:** Standard license usage

### SHAP Explanations

Top 4 features contributing to the risk score are provided with SHAP values:
- Negative SHAP values reduce risk
- Positive SHAP values increase risk

### Typosquatting Detection

For CRITICAL risk packages, the API suggests similar popular packages that might be the intended target.

### Intelligent Caching

- **Cache TTL:** 30 minutes
- **Automatic expiry:** Old cache entries are removed on access
- **Manual clear:** Use `/cache/clear` endpoint

### Rate Limiting

- **Limit:** 10 requests per 60 seconds per IP
- **Status:** 429 (Too Many Requests) when exceeded
- **Headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`

### CORS Support

All endpoints support Cross-Origin Resource Sharing. Requests from any origin are allowed.

---

## Examples

### Python Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Single package analysis
response = requests.post(
    f"{BASE_URL}/analyze",
    json={"package_name": "express"}
)
result = response.json()
print(f"Package: {result['package']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Score: {result['score']:.4f}")

# Batch analysis
response = requests.post(
    f"{BASE_URL}/batch",
    json={"packages": ["express", "react", "vue", "lodash"]}
)
results = response.json()
for pkg in results:
    print(f"{pkg['package']}: {pkg['risk_level']}")

# Get cache status
response = requests.get(f"{BASE_URL}/cache/size")
cache_info = response.json()
print(f"Cached packages: {cache_info['cached_packages']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Analyze single package
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"package_name":"express"}'

# Batch analysis
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"packages":["express","react","vue"]}'

# Clear cache
curl http://localhost:8000/cache/clear

# Check cache size
curl http://localhost:8000/cache/size
```

### JavaScript/Node.js Example

```javascript
const BASE_URL = "http://localhost:8000";

// Analyze single package
async function analyzePackage(packageName) {
  const response = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ package_name: packageName })
  });
  return response.json();
}

// Example usage
const result = await analyzePackage("express");
console.log(`${result.package}: ${result.risk_level}`);
```

---

## Rate Limiting & Caching

### Rate Limiting Behavior

```
Max requests: 10 per 60 seconds per IP address

When limit exceeded:
- Status: 429 Too Many Requests
- Response: {"detail": "Rate limit exceeded: 10 requests per 60 seconds"}
- Headers: X-RateLimit-Limit: 10, X-RateLimit-Remaining: 0
```

### Caching Behavior

```
TTL: 30 minutes per entry

Cache keys: Package name (case-sensitive)
Automatic cleanup: Expired entries are removed on access
Manual clear: GET /cache/clear endpoint

Cache effects:
- First request for "express": ~1-3 seconds (fetches & analyzes)
- Subsequent requests within 30 min: ~10-50ms (from cache)
```

### Performance Tips

1. **Use batch endpoint** for multiple packages — more efficient than individual requests
2. **Leverage caching** — analyze same packages repeatedly for instant results
3. **Check cache size** — monitor memory usage with `/cache/size`
4. **Batch timing** — analyze up to 20 packages together, then wait between batches

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  scope-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
```

### Environment Variables

Currently no environment variables are required. The API uses default paths for models and data.

---

## Support

For issues or questions:
1. Check the [README.md](README.md) for general project information
2. Visit `http://localhost:8000/docs` for interactive API documentation
3. Review error responses and HTTP status codes above

---

## License

See [LICENSE](LICENSE) in the project root.
