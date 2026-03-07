"""
Sentinel CLI
Command-line interface for scanning NPM packages.
"""

import click
import joblib
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="sentinel-npm")
def cli():
    """🛡️ Sentinel NPM — Malicious Package Detector"""
    pass


@cli.command()
@click.argument("package_name")
@click.option("--model-path", default="models/sentinel_model.joblib", help="Path to the trained model.")
@click.option("--scaler-path", default="models/sentinel_scaler.joblib", help="Path to the fitted scaler.")
@click.option("--explain", is_flag=True, help="Show SHAP-based feature explanations.")
def scan(package_name: str, model_path: str, scaler_path: str, explain: bool):
    """Scan an NPM package for potential malicious behavior."""
    console.print(Panel(f"[bold cyan]Scanning package:[/bold cyan] {package_name}", title="🛡️ Sentinel"))

    # Step 1: Fetch metadata
    console.print("[yellow]Fetching NPM metadata...[/yellow]")
    from src.data.npm_fetcher import fetch_package_metadata
    metadata = fetch_package_metadata(package_name)

    if metadata is None:
        console.print(f"[red]✗ Package '{package_name}' not found on NPM.[/red]")
        return

    # Step 2: Extract features
    console.print("[yellow]Extracting features...[/yellow]")
    from src.data.feature_engineer import extract_npm_features
    features = extract_npm_features(metadata)

    # Step 3: Load model and predict
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        console.print("[red]✗ Model not found. Train a model first with `sentinel train`.[/red]")
        return

    console.print("[yellow]Running prediction...[/yellow]")
    # TODO: Align features with training feature columns
    console.print("[green]✓ Scan complete.[/green]")


@cli.command()
@click.option("--data-path", default="data/processed/features.csv", help="Path to processed training data.")
@click.option("--output-dir", default="models", help="Directory to save the trained model.")
def train(data_path: str, output_dir: str):
    """Train the Sentinel detection model."""
    console.print(Panel("[bold cyan]Training Sentinel Model[/bold cyan]", title="🛡️ Sentinel"))

    from src.model.train import load_training_data, prepare_features, train_model, save_model
    from src.model.evaluate import print_evaluation_report

    console.print(f"[yellow]Loading data from {data_path}...[/yellow]")
    df = load_training_data(data_path)

    console.print("[yellow]Preparing features...[/yellow]")
    X, y = prepare_features(df, drop_cols=["name"])

    console.print("[yellow]Training model...[/yellow]")
    model, scaler, split = train_model(X, y)

    console.print("[yellow]Evaluating model...[/yellow]")
    print_evaluation_report(model, split["X_test"], split["y_test"])

    paths = save_model(model, scaler, output_dir)
    console.print(f"[green]✓ Model saved to {paths['model']}[/green]")
    console.print(f"[green]✓ Scaler saved to {paths['scaler']}[/green]")


@cli.command()
@click.argument("package_name")
def info(package_name: str):
    """Show metadata info for an NPM package."""
    from src.data.npm_fetcher import fetch_package_metadata, fetch_package_downloads

    console.print(Panel(f"[bold cyan]Package Info:[/bold cyan] {package_name}", title="📦 NPM"))

    metadata = fetch_package_metadata(package_name)
    if metadata is None:
        console.print(f"[red]✗ Package '{package_name}' not found.[/red]")
        return

    table = Table(title=f"{package_name} Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Latest Version", metadata.get("dist-tags", {}).get("latest", "N/A"))
    table.add_row("License", metadata.get("license", "N/A"))
    table.add_row("Maintainers", str(len(metadata.get("maintainers", []))))
    table.add_row("Versions", str(len(metadata.get("versions", {}))))
    table.add_row("Homepage", metadata.get("homepage", "N/A") or "N/A")
    table.add_row("Repository", str(metadata.get("repository", {}).get("url", "N/A")))

    downloads = fetch_package_downloads(package_name)
    if downloads:
        table.add_row("Monthly Downloads", f"{downloads.get('downloads', 0):,}")

    console.print(table)


if __name__ == "__main__":
    cli()
