
import argparse
import json
import sys
import os
import joblib
import pandas as pd
import numpy as np
import difflib
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel

# Add src to path if needed
sys.path.append(os.getcwd())

from src.data.npm_fetcher import fetch_npm_raw
from src.data.github_fetcher import fetch_github_stats
from src.data.feature_engineer import engineer_features
from src.model.explain import get_shap_explainer, explain_single_prediction
from src.cli.output import format_result

console = Console()

VERSION = "1.0.0"

class PackageNotFoundError(Exception):
    """Raised when a package is not found on npm."""
    pass

class SentinelEngine:
    def __init__(self, 
                 model_path="models/scope_model.joblib", 
                 scaler_path="models/scope_scaler.joblib",
                 popular_pkgs_path="data/healthy_packages.txt"):
        # Load model and scaler
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model ({model_path}) or scaler ({scaler_path}) not found. Please run training first.")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # XGBoost/RF models might store feature_names. 
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = self.model.feature_names_in_.tolist()
        else:
            self.feature_names = [
                "days_since_created", "days_since_last_update", "num_versions", 
                "release_velocity", "num_maintainers", "has_postinstall", "description_length", 
                "license_is_standard", "has_github_repo", "stargazers_count", "forks_count", 
                "open_issues_count", "subscribers_count", "contributor_count", "days_since_last_commit"
            ]
        
        self.explainer = get_shap_explainer(self.model)
        
        # Load popular packages for typosquatting detection
        self.popular_packages = []
        if os.path.exists(popular_pkgs_path):
            with open(popular_pkgs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.popular_packages.append(line)
        else:
            console.print(f"[yellow]Warning: Popular packages list not found at {popular_pkgs_path}. Typosquatting detection disabled.[/yellow]")

    def suggest_intended_package(self, package_name):
        """Find if a package name is suspiciously similar to a popular one."""
        if not self.popular_packages or package_name in self.popular_packages:
            return None
            
        matches = difflib.get_close_matches(package_name, self.popular_packages, n=1, cutoff=0.8)
        if matches:
            suggested = matches[0]
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, package_name, suggested).ratio()
            return {"name": suggested, "similarity": similarity}
        return None

    def analyze(self, package_name, skip_suggestion=False):
        """Analyze a single package."""
        warnings = []
        try:
            # 1. Fetch NPM
            npm_raw = fetch_npm_raw(package_name)
            if not npm_raw:
                raise PackageNotFoundError(f"Package '{package_name}' not found on npm.")
            
            # 2. Fetch GitHub
            repo_field = npm_raw.get("repository")
            if not repo_field:
                warnings.append("No repository link found (potential risk factor).")
                
            github_stats = fetch_github_stats(repo_field)
            if not github_stats.get("has_github_repo", 0) and repo_field:
                warnings.append("GitHub data unavailable (rate-limited or not found). Proceeding with zeros.")
            
            # 3. Engineer features
            features = engineer_features(npm_raw, github_stats)
            
            # 4. Preprocess
            X_df = pd.DataFrame([features])
            X_data = X_df[self.feature_names]
            X_scaled = self.scaler.transform(X_data)
            
            # 5. Predict
            score = float(self.model.predict_proba(X_scaled)[0, 1])
            
            # 6. Explain
            explanations = explain_single_prediction(self.explainer, X_scaled, self.feature_names)
            
            result = {
                "package": package_name,
                "score": score,
                "risk_level": self._get_risk_level(score),
                "features": features,
                "explanations": explanations,
                "warnings": warnings
            }

            # 7. Typosquatting Check
            if not skip_suggestion and result["risk_level"] in ["HIGH", "CRITICAL"]:
                suggestion = self.suggest_intended_package(package_name)
                if suggestion:
                    # Fetch score for the suggested (popular) package for comparison
                    # skip_suggestion=True to avoid infinite recursion
                    s_result = self.analyze(suggestion["name"], skip_suggestion=True)
                    if "score" in s_result:
                        suggestion["score"] = s_result["score"]
                        result["suggestion"] = suggestion

            return result
        except PackageNotFoundError as e:
            return {"package": package_name, "error": str(e), "status": "NOT_FOUND"}
        except Exception as e:
            return {"package": package_name, "error": str(e), "status": "ERROR"}


    def _get_risk_level(self, score):
        if score < 0.2: return "HEALTHY"
        if score < 0.5: return "MEDIUM"
        if score < 0.8: return "HIGH"
        return "CRITICAL"

    def analyze_many(self, package_names):
        """Analyze a list of packages with progress bar."""
        results = []
        for pkg in track(package_names, description="Analyzing packages..."):
            results.append(self.analyze(pkg))
        return results

def parse_package_json(filepath):
    """Parse dependencies and devDependencies from package.json."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            return list(set(list(deps.keys()) + list(dev_deps.keys())))
    except Exception as e:
        console.print(f"[bold red]Error parsing package.json:[/bold red] {e}")
        return []

def parse_requirements_txt(filepath):
    """Parse a simple requirements.txt (packages on separate lines)."""
    try:
        packages = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                pkg = line.strip()
                if pkg and not pkg.startswith("#"):
                    if "==" in pkg: pkg = pkg.split("==")[0].strip()
                    elif ">=" in pkg: pkg = pkg.split(">=")[0].strip()
                    elif "<=" in pkg: pkg = pkg.split("<=")[0].strip()
                    packages.append(pkg)
        return packages
    except Exception as e:
        console.print(f"[bold red]Error parsing requirements.txt:[/bold red] {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel: AI-powered NPM Package Security Scoring Tool",
        epilog="Example: sentinel check lodash"
    )
    parser.add_argument("--version", action="version", version=f"Sentinel {VERSION}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    check_parser = subparsers.add_parser("check", help="Score a single package from the NPM registry")
    check_parser.add_argument("package", help="Name of the NPM package to analyze (e.g., 'express', 'lodash')")
    check_parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON format")

    batch_parser = subparsers.add_parser("batch", help="Score multiple packages from a project file")
    batch_parser.add_argument("file", help="Path to package.json or requirements.txt style file")
    batch_parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON format")
    batch_parser.add_argument("--fail-on-high", action="store_true", help="Exit with code 1 if any package score exceeds 0.80")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        engine = SentinelEngine()
    except Exception as e:
        console.print(f"[bold red]Error initializing engine:[/bold red] {e}")
        sys.exit(1)

    if args.command == "check":
        result = engine.analyze(args.package)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            console.print(format_result(result))
        
        if "error" in result:
            sys.exit(1 if result.get("status") != "NOT_FOUND" else 0)

    elif args.command == "batch":
        packages = []
        if args.file.endswith("package.json"):
            packages = parse_package_json(args.file)
        else:
            packages = parse_requirements_txt(args.file)
        
        if not packages:
            console.print("[bold red]No packages found in file.[/bold red]")
            sys.exit(1)
        
        results = engine.analyze_many(packages)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            table = Table(title="Batch Analysis Summary")
            table.add_column("Package", style="cyan")
            table.add_column("Risk Level", justify="center")
            table.add_column("Score", justify="right")
            
            summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "HEALTHY": 0, "ERRORS": 0}
            
            for res in results:
                if "error" in res:
                    table.add_row(res.get('package', 'Unknown'), "[red]ERROR[/red]", "N/A")
                    summary["ERRORS"] += 1
                else:
                    risk = res['risk_level']
                    score = res['score']
                    color = "green" if risk == "HEALTHY" else "yellow" if risk == "MEDIUM" else "red" if risk == "HIGH" else "bold red"
                    table.add_row(res['package'], f"[{color}]{risk}[/{color}]", f"{score*100:.1f}%")
                    summary[risk] += 1
            
            console.print(table)
            
            summary_line = f"Scanned {len(results)} packages. "
            if summary["CRITICAL"] > 0: summary_line += f"[bold red]{summary['CRITICAL']} CRITICAL[/bold red], "
            if summary["HIGH"] > 0: summary_line += f"[red]{summary['HIGH']} HIGH[/red], "
            if summary["MEDIUM"] > 0: summary_line += f"[yellow]{summary['MEDIUM']} MEDIUM[/yellow], "
            if summary["HEALTHY"] > 0: summary_line += f"[green]{summary['HEALTHY']} HEALTHY[/green]"
            if summary["ERRORS"] > 0: summary_line += f", [bold white on red]{summary['ERRORS']} ERRORS[/bold white on red]"
            
            console.print(Panel(summary_line.strip(", ")))

        if args.fail_on_high:
            for res in results:
                if res.get('score', 0) > 0.8:
                    console.print("\n[bold red]FATAL: Critical risk packages found![/bold red]")
                    sys.exit(1)

if __name__ == "__main__":
    main()
