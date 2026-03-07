"""
Test script to verify all required libraries are installed correctly.
"""

import importlib
import sys

libraries = {
    # (import name, pip/requirements name)
    "requests":      "requests",
    "pandas":        "pandas",
    "numpy":         "numpy",
    "sklearn":       "scikit-learn",
    "xgboost":       "xgboost",
    "shap":          "shap",
    "fastapi":       "fastapi",
    "uvicorn":       "uvicorn",
    "click":         "click",
    "rich":          "rich",
}

passed = 0
failed = 0

print("=" * 50)
print("  Library Import Verification")
print("=" * 50)

for import_name, pip_name in libraries.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [PASS]  {pip_name:<15} (v{version})")
        passed += 1
    except ImportError as e:
        print(f"  [FAIL]  {pip_name:<15} -- MISSING ({e})")
        failed += 1

print("=" * 50)
print(f"  Results: {passed} passed, {failed} failed")
print("=" * 50)

if failed:
    print("\n  Some libraries are missing. Install them with:")
    print("    pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n  All libraries installed correctly!")
    sys.exit(0)
