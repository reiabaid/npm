"""SCOPE FastAPI REST API — NPM Package Security Analysis Service."""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.cli.sentinel import ScopeEngine
from src.data.osv_fetcher import query_osv

# ============================================================================
# Global State
# ============================================================================

engine: Optional[ScopeEngine] = None
rate_limit_tracker: Dict[str, List[datetime]] = defaultdict(list)

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds
CACHE_TTL = 30 * 60  # 30 minutes in seconds
CACHE_FILE = "/tmp/scope_cache.json"

import json as _json
import threading as _threading
_cache_lock = _threading.Lock()

def _load_cache() -> Dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return _json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(data: Dict):
    try:
        with open(CACHE_FILE, "w") as f:
            _json.dump(data, f)
    except Exception:
        pass

def _cache_get(key: str):
    with _cache_lock:
        store = _load_cache()
        entry = store.get(key)
        if not entry:
            return None
        if datetime.fromisoformat(entry["expires"]) < datetime.utcnow():
            store.pop(key, None)
            _save_cache(store)
            return None
        return entry["value"]

def _cache_set(key: str, value: Dict):
    with _cache_lock:
        store = _load_cache()
        store[key] = {
            "value": value,
            "expires": (datetime.utcnow() + timedelta(seconds=CACHE_TTL)).isoformat(),
        }
        _save_cache(store)
BATCH_MAX_SIZE = 20

# ============================================================================
# Pydantic Models
# ============================================================================

class PackageRequest(BaseModel):
    """Request model for single package analysis."""
    package_name: str = Field(..., min_length=1, description="Name of the NPM package to analyze")

class ShapFactor(BaseModel):
    """SHAP explanation factor."""
    feature: str
    shap_value: float
    description: Optional[str] = None

class AnalysisResult(BaseModel):
    """Response model for package analysis."""
    package: str
    score: float = Field(..., ge=0.0, le=1.0, description="Risk score from 0 (healthy) to 1 (critical)")
    risk_level: str = Field(..., description="HEALTHY, MEDIUM, HIGH, or CRITICAL")
    features: Optional[Dict] = None
    explanations: List[ShapFactor] = []
    warnings: List[str] = []
    suggestion: Optional[Dict] = None
    llm_verdict: Optional[str] = None
    error: Optional[str] = None
    status: Optional[str] = None

class BatchRequest(BaseModel):
    """Request model for batch analysis."""
    packages: List[str] = Field(..., min_length=1, max_length=BATCH_MAX_SIZE, description="List of package names (max 20)")

class DashboardRequest(BaseModel):
    """Request model for dependency dashboard scan."""
    package_json: dict

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str

# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="SCOPE API",
    description="AI-powered NPM Package Security Scoring Tool",
    version="1.0.0",
    docs_url="/docs",
)

# ============================================================================
# Middleware: CORS
# ============================================================================

_raw_origin = os.environ.get("ALLOWED_ORIGIN", "")
_allowed_origins = (
    [o.strip() for o in _raw_origin.split(",") if o.strip()]
    if _raw_origin
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================================
# Middleware: Rate Limiting
# ============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware: 10 requests per 60 seconds per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now()

    if client_ip in rate_limit_tracker:
        rate_limit_tracker[client_ip] = [
            req_time for req_time in rate_limit_tracker[client_ip]
            if (now - req_time).total_seconds() < RATE_LIMIT_WINDOW
        ]

    if len(rate_limit_tracker[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests per 60 seconds")

    rate_limit_tracker[client_ip].append(now)

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT_REQUESTS - len(rate_limit_tracker[client_ip]))
    return response

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load the SCOPE engine once at startup."""
    global engine
    try:
        engine = ScopeEngine()
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to load SCOPE engine: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global engine, rate_limit_tracker
    engine = None
    rate_limit_tracker.clear()

# ============================================================================
# Helper Functions
# ============================================================================

def _get_cached_result(package_name: str) -> Optional[Dict]:
    return _cache_get(package_name)

def _cache_result(package_name: str, result: Dict):
    _cache_set(package_name, result)

def _shap_factors_to_model(explanations: List[Dict]) -> List[ShapFactor]:
    """Convert SHAP explanations to Pydantic models."""
    return [
        ShapFactor(
            feature=exp.get("feature", ""),
            shap_value=exp.get("shap_value", 0.0),
            description=None
        )
        for exp in explanations[:4]
    ]

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=(engine is not None),
        timestamp=datetime.now().isoformat()
    )

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_package(request: PackageRequest):
    """
    Analyze a single NPM package for security risks.

    - **package_name**: Name of the NPM package to analyze (e.g., 'express', 'lodash')

    Returns risk score, level, and SHAP-based feature explanations.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="SCOPE engine not initialized")

    package_name = request.package_name.strip()

    cached = _get_cached_result(package_name)
    if cached:
        return AnalysisResult(**cached)

    try:
        result = await asyncio.to_thread(engine.analyze, package_name)

        if "error" in result:
            if result.get("status") == "NOT_FOUND":
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Package '{package_name}' not found on npm",
                             "suggestion": result.get("suggestion")},
                )
            else:
                raise HTTPException(status_code=503, detail=f"Error analyzing package: {result['error']}")

        response_dict = {
            "package": result["package"],
            "score": result["score"],
            "risk_level": result["risk_level"],
            "features": result.get("features"),
            "explanations": _shap_factors_to_model(result.get("explanations", [])),
            "warnings": result.get("warnings", []),
            "suggestion": result.get("suggestion"),
            "llm_verdict": result.get("llm_verdict"),
        }

        _cache_result(package_name, response_dict)
        return AnalysisResult(**response_dict)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch", response_model=List[AnalysisResult])
async def analyze_batch(request: BatchRequest):
    """
    Analyze multiple NPM packages in a batch.

    - **packages**: List of package names (max 20)

    Returns list of analysis results.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="SCOPE engine not initialized")

    if len(request.packages) > BATCH_MAX_SIZE:
        raise HTTPException(status_code=422, detail=f"Too many packages. Maximum is {BATCH_MAX_SIZE}")

    unique_packages = []
    seen = set()
    for pkg in request.packages:
        pkg_clean = pkg.strip()
        if pkg_clean and pkg_clean not in seen:
            unique_packages.append(pkg_clean)
            seen.add(pkg_clean)

    if not unique_packages:
        raise HTTPException(status_code=422, detail="No valid packages provided")

    try:
        results = await asyncio.to_thread(engine.analyze_many, unique_packages)

        analysis_results = []
        for result in results:
            if "error" not in result:
                response_dict = {
                    "package": result["package"],
                    "score": result["score"],
                    "risk_level": result["risk_level"],
                    "features": result.get("features"),
                    "explanations": _shap_factors_to_model(result.get("explanations", [])),
                    "warnings": result.get("warnings", []),
                    "suggestion": result.get("suggestion"),
                    "llm_verdict": result.get("llm_verdict"),
                }
                _cache_result(result["package"], response_dict)
            else:
                response_dict = {
                    "package": result["package"],
                    "error": result.get("error"),
                    "status": result.get("status"),
                }

            analysis_results.append(AnalysisResult(**response_dict))

        return analysis_results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/dashboard/scan")
async def dashboard_scan(req: DashboardRequest):
    """Scan all dependencies in a package.json for ML risk and known CVEs."""
    if not engine:
        raise HTTPException(status_code=503, detail="SCOPE engine not initialized")

    _engine = engine

    deps: Dict[str, str] = {}
    deps.update(req.package_json.get("dependencies", {}))
    deps.update(req.package_json.get("devDependencies", {}))

    async def scan_one(name: str, version_str: str):
        version = version_str.lstrip("^~>=<")
        ml, vulns = await asyncio.gather(
            asyncio.to_thread(_engine.analyze, name),
            asyncio.to_thread(query_osv, name, version),
        )
        return {
            "package":    name,
            "version":    version,
            "risk_level": ml.get("risk_level", "UNKNOWN"),
            "score":      ml.get("score"),
            "cves":       [{"id": v["id"], "summary": v.get("summary", "")} for v in vulns],
        }

    results = list(await asyncio.gather(*[scan_one(n, v) for n, v in deps.items()]))
    results.sort(key=lambda x: (len(x["cves"]), x.get("score") or 0), reverse=True)
    return results

@app.get("/cache/clear")
async def clear_cache():
    """Clear the response cache."""
    with _cache_lock:
        _save_cache({})
    return {"status": "ok", "message": "Cache cleared"}

@app.get("/cache/size")
async def get_cache_size():
    """Get current cache size."""
    with _cache_lock:
        store = _load_cache()
    return {"cached_packages": len(store), "cache_ttl_seconds": CACHE_TTL}

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import os
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info",
    )
