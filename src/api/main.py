"""SCOPE FastAPI REST API — NPM Package Security Analysis Service."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.cli.sentinel import ScopeEngine

# ============================================================================
# Global State
# ============================================================================

engine: Optional[ScopeEngine] = None
response_cache: Dict[str, Dict] = {}
cache_expiry: Dict[str, datetime] = {}
rate_limit_tracker: Dict[str, List[datetime]] = defaultdict(list)

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds
CACHE_TTL = 30 * 60  # 30 minutes in seconds
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
    error: Optional[str] = None
    status: Optional[str] = None

class BatchRequest(BaseModel):
    """Request model for batch analysis."""
    packages: List[str] = Field(..., min_items=1, max_items=BATCH_MAX_SIZE, description="List of package names (max 20)")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Middleware: Rate Limiting
# ============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware: 10 requests per 60 seconds per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now()
    
    # Clean up old requests outside the window
    if client_ip in rate_limit_tracker:
        rate_limit_tracker[client_ip] = [
            req_time for req_time in rate_limit_tracker[client_ip]
            if (now - req_time).total_seconds() < RATE_LIMIT_WINDOW
        ]
    
    # Check rate limit
    if len(rate_limit_tracker[client_ip]) >= RATE_LIMIT_REQUESTS:
        return HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests per 60 seconds")
    
    # Record this request
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
    global engine, response_cache, cache_expiry, rate_limit_tracker
    engine = None
    response_cache.clear()
    cache_expiry.clear()
    rate_limit_tracker.clear()

# ============================================================================
# Helper Functions
# ============================================================================

def _get_cached_result(package_name: str) -> Optional[Dict]:
    """Retrieve cached result if it exists and hasn't expired."""
    if package_name in response_cache:
        if package_name in cache_expiry:
            if datetime.now() < cache_expiry[package_name]:
                return response_cache[package_name]
        # Remove expired cache
        response_cache.pop(package_name, None)
        cache_expiry.pop(package_name, None)
    return None

def _cache_result(package_name: str, result: Dict):
    """Cache a result with expiry time."""
    response_cache[package_name] = result
    cache_expiry[package_name] = datetime.now() + timedelta(seconds=CACHE_TTL)

def _shap_factors_to_model(explanations: List[Dict]) -> List[ShapFactor]:
    """Convert SHAP explanations to Pydantic models."""
    return [
        ShapFactor(
            feature=exp.get("feature", ""),
            shap_value=exp.get("shap_value", 0.0),
            description=None
        )
        for exp in explanations[:4]  # Top 4 factors
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
    
    # Check cache first
    cached = _get_cached_result(package_name)
    if cached:
        return AnalysisResult(**cached)
    
    # Run analysis in thread pool to avoid blocking event loop
    try:
        result = await asyncio.to_thread(engine.analyze, package_name)
        
        # Convert to AnalysisResult format
        if "error" in result:
            if result.get("status") == "NOT_FOUND":
                raise HTTPException(status_code=404, detail=f"Package '{package_name}' not found on npm")
            else:
                raise HTTPException(status_code=503, detail=f"Error analyzing package: {result['error']}")
        
        # Format response
        response_dict = {
            "package": result["package"],
            "score": result["score"],
            "risk_level": result["risk_level"],
            "features": result.get("features"),
            "explanations": _shap_factors_to_model(result.get("explanations", [])),
            "warnings": result.get("warnings", []),
            "suggestion": result.get("suggestion"),
        }
        
        # Cache the successful result
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
    
    # Remove duplicates while preserving order
    unique_packages = []
    seen = set()
    for pkg in request.packages:
        pkg_clean = pkg.strip()
        if pkg_clean and pkg_clean not in seen:
            unique_packages.append(pkg_clean)
            seen.add(pkg_clean)
    
    if not unique_packages:
        raise HTTPException(status_code=422, detail="No valid packages provided")
    
    # Run batch analysis in thread pool
    try:
        results = await asyncio.to_thread(engine.analyze_many, unique_packages)
        
        # Format results
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
                }
                # Cache each result
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

@app.get("/cache/clear")
async def clear_cache():
    """Clear the response cache."""
    global response_cache, cache_expiry
    response_cache.clear()
    cache_expiry.clear()
    return {"status": "ok", "message": "Cache cleared"}

@app.get("/cache/size")
async def get_cache_size():
    """Get current cache size."""
    return {"cached_packages": len(response_cache), "cache_ttl_seconds": CACHE_TTL}

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
