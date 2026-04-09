
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress
from rich.console import Group

def get_risk_color(score: float) -> str:
    if score < 0.3:
        return "green"
    elif score < 0.6:
        return "yellow"
    elif score < 0.8:
        return "orange3"
    else:
        return "red"

def build_risk_bar(score: float, width: int = 20) -> Text:
    """Build a visual risk bar using ASCII characters."""
    filled_length = int(score * width)
    bar_text = "#" * filled_length + "-" * (width - filled_length)
    color = get_risk_color(score)
    return Text(f"[{bar_text}]", style=color)

def format_result(result: dict) -> Panel:
    """Format a single package analysis result into a rich Panel."""
    if "error" in result:
        return Panel(
            Text(f"X {result.get('package', 'Unknown')}: {result['error']}", style="bold red"),
            title="[bold red]Analysis Error[/bold red]",
            border_style="red"
        )

    package_name = result["package"]
    score = result["score"]
    risk_level = result["risk_level"]
    color = get_risk_color(score)
    
    # 1. Header Information
    display = Text()
    display.append("Package: ", style="bold")
    display.append(f"{package_name}\n", style="cyan")
    
    display.append("Risk Score: ", style="bold")
    display.append(f"{score*100:.1f}%", style=f"bold {color}")
    display.append(f" ({risk_level})\n", style=color)
    
    # 2. Risk Bar
    display.append("Risk Bar:    ", style="bold")
    display.append(build_risk_bar(score))
    display.append("\n")

    # 3. Warnings
    if result.get("warnings"):
        display.append("\nWarnings:\n", style="bold yellow")
        for warning in result["warnings"]:
            display.append(f" ! {warning}\n", style="yellow")

    # 4. Typosquatting
    render_items = [display]
    
    if result.get("suggestion"):
        s = result["suggestion"]
        typo_text = Text("\n POSSIBLE TYPOSQUATTING DETECTED \n", style="bold white on red", justify="center")
        
        suggest_text = Text()
        suggest_text.append(f"Did you mean: ", style="bold")
        suggest_text.append(f"{s['name']}", style="bold cyan")
        suggest_text.append(f"? (", style="bold")
        suggest_text.append(f"{s['score']*100:.1f}% risk", style=f"bold {get_risk_color(s['score'])}")
        suggest_text.append(")\n", style="bold")
        suggest_text.append(f"Similarity: {s['similarity']:.1%} match with popular package.\n", style="italic")
        
        render_items.append(typo_text)
        render_items.append(suggest_text)

    # 5. SHAP Factors Table
    factors_table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold magenta")
    factors_table.add_column("Risk Factor")
    factors_table.add_column("Impact", justify="center")
    factors_table.add_column("Weight", justify="right")

    for e in result.get("explanations", [])[:4]:
        val = e["shap_value"]
        direction = "+ HIGHER" if val > 0 else "- LOWER"
        dir_style = "bold red" if val > 0 else "bold green"
        
        factors_table.add_row(
            e["feature"],
            Text(direction, style=dir_style),
            f"{abs(val):.4f}"
        )

    render_items.append(Text("\nTop Risk Contributors:", style="bold underline"))
    render_items.append(factors_table)

    return Panel(
        Group(*render_items),
        title=f"[bold white]Sentinel Analysis[/bold white]",
        border_style=color,
        padding=(1, 2)
    )
