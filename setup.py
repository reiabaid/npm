from setuptools import setup, find_packages

setup(
    name="scope-npm",
    version="1.0.0",
    description="AI-powered NPM package security scoring and risk analysis tool",
    author="SCOPE Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.43.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "rich>=13.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "scope=src.cli.scope:main",
        ]
    },
    include_package_data=True,
)
