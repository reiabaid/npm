"""
SCOPE API
REST API for scanning NPM packages.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="SCOPE API",
    description="🔬 SCOPE — Security Check for Open-source Package Ecosystems",
    version="0.1.0",
)

# ── Models ────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    package_name: str

class ScanResponse(BaseModel):
    package_name: str
    risk_score: float
    is_malicious: bool
    features: dict
    explanation: list | None = None

class PackageInfo(BaseModel):
    name: str
    latest_version: str
    license: str | None
    maintainer_count: int
    version_count: int
    monthly_downloads: int | None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_exists = os.path.exists("models/scope_model.joblib")
    return HealthResponse(status="healthy", model_loaded=model_exists)


@app.post("/scan", response_model=ScanResponse)
async def scan_package(request: ScanRequest):
    """Scan an NPM package for malicious indicators."""
    from src.data.npm_fetcher import fetch_package_metadata
    from src.data.feature_engineer import extract_npm_features

    metadata = fetch_package_metadata(request.package_name)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Package '{request.package_name}' not found on NPM.")

    features = extract_npm_features(metadata)

    # Load model if available
    model_path = "models/scope_model.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet. Train a model first.")

    model = joblib.load(model_path)
    scaler = joblib.load("models/scope_scaler.joblib")

    # TODO: Align features and predict
    return ScanResponse(
        package_name=request.package_name,
        risk_score=0.0,
        is_malicious=False,
        features=features,
        explanation=None,
    )


@app.get("/info/{package_name}", response_model=PackageInfo)
async def get_package_info(package_name: str):
    """Get metadata information for an NPM package."""
    from src.data.npm_fetcher import fetch_package_metadata, fetch_package_downloads

    metadata = fetch_package_metadata(package_name)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Package '{package_name}' not found on NPM.")

    downloads_data = fetch_package_downloads(package_name)

    return PackageInfo(
        name=metadata.get("name", package_name),
        latest_version=metadata.get("dist-tags", {}).get("latest", "unknown"),
        license=metadata.get("license"),
        maintainer_count=len(metadata.get("maintainers", [])),
        version_count=len(metadata.get("versions", {})),
        monthly_downloads=downloads_data.get("downloads") if downloads_data else None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
