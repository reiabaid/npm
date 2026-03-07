"""
Manual test: fetch, save, and parse 5 NPM packages.
Prints a readable summary of the extracted fields.
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.npm_fetcher import fetch_and_save

PACKAGES = ["lodash", "axios", "express", "react", "left-pad"]

print("=" * 60)
print("  NPM Fetcher -- Manual Test (5 packages)")
print("=" * 60)

for pkg in PACKAGES:
    print(f"\n{'-' * 60}")
    print(f"  Fetching: {pkg}")
    print(f"{'-' * 60}")

    parsed = fetch_and_save(pkg)
    if parsed is None:
        print(f"  [SKIP] Could not fetch '{pkg}'.\n")
        continue

    desc = parsed['description']
    desc_short = (desc[:80] + "...") if desc and len(desc) > 80 else (desc or "N/A")

    print(f"  Name:            {parsed['name']}")
    print(f"  Description:     {desc_short}")
    print(f"  Created:         {parsed['created']}")
    print(f"  Modified:        {parsed['modified']}")
    print(f"  Latest version:  {parsed['latest_version']}")
    print(f"  Total versions:  {len(parsed['version_timestamps'])}")
    print(f"  Maintainers:     {', '.join(m.get('name','?') for m in parsed['maintainers'])}")
    kw = parsed['keywords']
    print(f"  Keywords:        {', '.join(kw[:10]) if kw else 'None'}")
    print(f"  Scripts (latest): {parsed['scripts'] if parsed['scripts'] else 'None'}")

print(f"\n{'=' * 60}")
print("  Done! Check data/raw/ for the saved JSON files.")
print(f"{'=' * 60}")
