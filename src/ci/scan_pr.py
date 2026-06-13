"""CI scan script — reads package.json, runs SCOPE, writes a markdown PR comment.

Exit code 0: all packages HEALTHY or MEDIUM
Exit code 1: at least one HIGH or CRITICAL package found
"""

import json
import os
import sys

from src.cli.sentinel import ScopeEngine

RISK_EMOJI = {
    "HEALTHY":  "✅",
    "MEDIUM":   "⚠️",
    "HIGH":     "🔴",
    "CRITICAL": "🚨",
}

COMMENT_FILE = "scan_comment.md"


def load_deps(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        pkg = json.load(f)
    deps = {}
    deps.update(pkg.get("dependencies", {}))
    deps.update(pkg.get("devDependencies", {}))
    return deps


def format_comment(results: list) -> str:
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "HEALTHY": 0}
    for r in results:
        level = r.get("risk_level", "UNKNOWN").upper()
        if level in counts:
            counts[level] += 1

    # Summary line
    parts = []
    if counts["CRITICAL"]: parts.append(f"🚨 {counts['CRITICAL']} critical")
    if counts["HIGH"]:     parts.append(f"🔴 {counts['HIGH']} high")
    if counts["MEDIUM"]:   parts.append(f"⚠️ {counts['MEDIUM']} medium")
    if counts["HEALTHY"]:  parts.append(f"✅ {counts['HEALTHY']} healthy")
    summary = " · ".join(parts) if parts else "No packages scanned"

    lines = [
        "## SCOPE Dependency Scan",
        "",
        f"**{summary}** across {len(results)} packages",
        "",
        "| Package | Risk | Score | Top Signal |",
        "|---------|------|-------|------------|",
    ]

    for r in sorted(results, key=lambda x: x.get("score") or 0, reverse=True):
        level = r.get("risk_level", "UNKNOWN")
        emoji = RISK_EMOJI.get(level, "❓")
        score = f"{(r.get('score') or 0) * 100:.0f}%"
        top = r.get("explanations", [{}])[0].get("feature", "—") if r.get("explanations") else "—"
        lines.append(f"| `{r['package']}` | {emoji} {level} | {score} | {top} |")

    lines += [
        "",
        "<sub>Powered by [SCOPE](https://github.com/reiabaid/npm) — "
        "explainable npm security scanner</sub>",
    ]
    return "\n".join(lines)


def main():
    pkg_path = "package.json"
    if not os.path.exists(pkg_path):
        print("No package.json found — skipping scan.")
        sys.exit(0)

    deps = load_deps(pkg_path)
    if not deps:
        print("No dependencies found in package.json.")
        sys.exit(0)

    print(f"Scanning {len(deps)} packages...")
    engine = ScopeEngine()
    results = engine.analyze_many(list(deps.keys()))

    comment = format_comment(results)

    with open(COMMENT_FILE, "w", encoding="utf-8") as f:
        f.write(comment)

    print(comment)

    has_high_risk = any(
        r.get("risk_level") in ("HIGH", "CRITICAL") for r in results
    )
    sys.exit(1 if has_high_risk else 0)


if __name__ == "__main__":
    main()
