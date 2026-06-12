
import argparse
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()
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

from src.data.npm_fetcher import fetch_npm_raw, fetch_package_downloads, fetch_maintainer_age
from src.data.github_fetcher import fetch_github_stats
from src.data.feature_engineer import engineer_features
from src.model.explain import get_shap_explainer, explain_single_prediction
from src.model.llm_review import get_llm_verdict
from src.cli.output import format_result
from src.cli.cache import ScopeCache
from src.cli.config import ScopeConfig

console = Console()

VERSION = "1.0.0"

# Feature names must match what the preprocessor was trained on
_FEATURE_NAMES = [
    "days_since_created", "days_since_last_update", "num_versions",
    "release_velocity", "num_maintainers", "description_length",
    "weekly_downloads", "typosquat_min_distance", "script_suspicion_score",
    "maintainer_min_account_age_days",
    "stargazers_count", "forks_count", "open_issues_count",
    "subscribers_count", "contributor_count", "days_since_last_commit",
    "has_any_install_hook", "license_is_standard", "has_github_repo",
]

DEFAULT_THRESHOLD = 0.5


class PackageNotFoundError(Exception):
    """Raised when a package is not found on npm."""
    pass

class ScopeEngine:
    def __init__(self,
                 model_path="models/scope_model.joblib",
                 preprocessor_path="models/scope_preprocessor.joblib",
                 threshold_path="models/scope_threshold.json",
                 popular_pkgs_path="data/healthy_packages.txt"):

        # Accept legacy scaler path as fallback so old models still work
        scaler_path = "models/scope_scaler.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run training first."
            )

        self.model = joblib.load(model_path)

        # Prefer new preprocessor (ColumnTransformer); fall back to legacy scaler
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            self._legacy_scaler = False
        elif os.path.exists(scaler_path):
            self.preprocessor = joblib.load(scaler_path)
            self._legacy_scaler = True
            console.print("[yellow]Using legacy scaler — retrain for full feature support.[/yellow]")
        else:
            raise FileNotFoundError(
                f"Neither {preprocessor_path} nor {scaler_path} found. Run training first."
            )

        # Load tuned classification threshold
        self.threshold = DEFAULT_THRESHOLD
        if os.path.exists(threshold_path):
            with open(threshold_path) as fh:
                self.threshold = json.load(fh).get("threshold", DEFAULT_THRESHOLD)

        # Load isotonic calibrator (optional — graceful fallback to raw proba)
        calibrator_path = "models/scope_calibrator.joblib"
        self.calibrator = joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None

        self.feature_names = _FEATURE_NAMES

        try:
            self.explainer = get_shap_explainer(self.model)
        except Exception:
            self.explainer = None

        # Load popular packages for typosquat detection
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

    def analyze(self, package_name, skip_suggestion=False, use_cache=True):
        """Analyze a single package."""
        if use_cache and not skip_suggestion:
            cached = ScopeCache.get(package_name)
            if cached:
                return cached

        warnings = []
        try:
            # 1. Fetch npm metadata
            npm_raw = fetch_npm_raw(package_name)
            if not npm_raw:
                raise PackageNotFoundError(f"Package '{package_name}' not found on npm.")

            # Extract install scripts early for LLM review later
            _latest = npm_raw.get("dist-tags", {}).get("latest", "")
            _all_scripts = npm_raw.get("versions", {}).get(_latest, {}).get("scripts", {})
            install_scripts = {k: v for k, v in _all_scripts.items()
                               if k in ("preinstall", "install", "postinstall")}

            # 2. Fetch GitHub metadata
            repo_field = npm_raw.get("repository")
            if not repo_field:
                warnings.append("No repository link found (potential risk factor).")
            github_stats = fetch_github_stats(repo_field)
            if not github_stats.get("has_github_repo", 0) and repo_field:
                warnings.append("GitHub data unavailable (rate-limited or not found). Proceeding with zeros.")

            # 3. Fetch download count and maintainer age (new signals)
            weekly_downloads = fetch_package_downloads(package_name)
            maintainers = npm_raw.get("maintainers", [])
            maintainer_min_age = fetch_maintainer_age(maintainers)

            # 4. Engineer features (pass popular packages for typosquat distance)
            features = engineer_features(
                npm_raw,
                github_stats,
                weekly_downloads=weekly_downloads,
                maintainer_min_age_days=maintainer_min_age,
                popular_names=self.popular_packages,
            )

            # 5. Preprocess
            X_df = pd.DataFrame([features])
            # Legacy path: old scaler expects original 15 features only
            if self._legacy_scaler:
                legacy_cols = [
                    "days_since_created", "days_since_last_update", "num_versions",
                    "release_velocity", "num_maintainers", "has_any_install_hook",
                    "description_length", "license_is_standard", "has_github_repo",
                    "stargazers_count", "forks_count", "open_issues_count",
                    "subscribers_count", "contributor_count", "days_since_last_commit",
                ]
                X_data = X_df.reindex(columns=legacy_cols, fill_value=0)
            else:
                X_data = X_df[self.feature_names]
            X_transformed = self.preprocessor.transform(X_data)

            # 6. Predict — apply isotonic calibrator if available
            raw_score = float(self.model.predict_proba(X_transformed)[0, 1])
            if self.calibrator is not None:
                score = float(self.calibrator.transform([raw_score])[0])
            else:
                score = raw_score

            # 7. Explain
            if self.explainer is not None:
                explanations = explain_single_prediction(
                    self.explainer, X_transformed, self.feature_names
                )
            else:
                explanations = []

            result = {
                "package":      package_name,
                "score":        score,
                "risk_level":   self._get_risk_level(score),
                "features":     features,
                "explanations": explanations,
                "warnings":     warnings,
            }

            # 8. LLM second-pass verdict (HIGH/CRITICAL only, skipped if no API key)
            if result["risk_level"] in ("HIGH", "CRITICAL"):
                verdict = get_llm_verdict(
                    package_name, features, explanations, install_scripts
                )
                if verdict:
                    result["llm_verdict"] = verdict

            # 9. Typosquatting check (post-hoc suggestion)
            if not skip_suggestion and result["risk_level"] in ["HIGH", "CRITICAL"]:
                suggestion = self.suggest_intended_package(package_name)
                if suggestion:
                    s_result = self.analyze(suggestion["name"], skip_suggestion=True)
                    if "score" in s_result:
                        suggestion["score"] = s_result["score"]
                        result["suggestion"] = suggestion

            ScopeCache.set(package_name, result)
            return result

        except PackageNotFoundError as e:
            result = {"package": package_name, "error": str(e), "status": "NOT_FOUND"}
            # Still try typosquat suggestion — common when someone misspells a name
            suggestion = self.suggest_intended_package(package_name)
            if suggestion:
                result["suggestion"] = suggestion
            return result
        except Exception as e:
            return {"package": package_name, "error": str(e), "status": "ERROR"}


    def _get_risk_level(self, score):
        t = self.threshold
        if score < t * 0.4:   return "HEALTHY"
        if score < t:         return "MEDIUM"
        if score < t + 0.3:   return "HIGH"
        return "CRITICAL"

    def analyze_many(self, package_names, use_cache=True):
        """Analyze a list of packages with progress bar."""
        results = []
        for pkg in track(package_names, description="Analyzing packages..."):
            results.append(self.analyze(pkg, use_cache=use_cache))
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
        description="SCOPE: AI-powered NPM Package Security Scoring Tool",
        epilog="Example: scope check lodash"
    )
    parser.add_argument("--version", action="version", version=f"SCOPE {VERSION}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    check_parser = subparsers.add_parser("check", help="Score a single package from the NPM registry")
    check_parser.add_argument("package", help="Name of the NPM package to analyze (e.g., 'express', 'lodash')")
    check_parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON format")
    check_parser.add_argument("--no-cache", action="store_true", help="Skip cache and always fetch fresh data")

    batch_parser = subparsers.add_parser("batch", help="Score multiple packages from a project file")
    batch_parser.add_argument("file", help="Path to package.json or requirements.txt style file")
    batch_parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON format")
    batch_parser.add_argument("--fail-on-high", action="store_true", help="Exit with code 1 if any package score exceeds 0.80")
    batch_parser.add_argument("--no-cache", action="store_true", help="Skip cache and always fetch fresh data")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        engine = ScopeEngine()
    except Exception as e:
        console.print(f"[bold red]Error initializing engine:[/bold red] {e}")
        sys.exit(1)

    if args.command == "check":
        result = engine.analyze(args.package, use_cache=not args.no_cache)
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
        
        results = engine.analyze_many(packages, use_cache=not args.no_cache)
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
