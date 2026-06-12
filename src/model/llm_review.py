"""LLM second-pass reviewer — called only on HIGH/CRITICAL packages.

Uses NVIDIA Build API (OpenAI-compatible). Set NVIDIA_API_KEY to enable.
If the key is unset the function returns None silently and SCOPE works normally.
"""

import os
from typing import Optional

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL    = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")


def get_llm_verdict(
    package_name: str,
    features: dict,
    explanations: list,
    install_scripts: dict | None = None,
) -> Optional[str]:
    """
    Query an NVIDIA-hosted LLM for a 2-sentence verdict on a flagged package.
    Returns None silently if NVIDIA_API_KEY is unset or on any error.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

        top_factors = "\n".join(
            f"  - {e['feature']}: {e['shap_value']:+.4f}"
            for e in (explanations or [])[:5]
        )

        script_section = ""
        if install_scripts:
            lines = [f"  {hook}: {cmd}" for hook, cmd in install_scripts.items()]
            if lines:
                script_section = "\nInstall scripts:\n" + "\n".join(lines)

        prompt = (
            f"You are a security analyst reviewing an npm package flagged as high risk.\n\n"
            f"Package: {package_name}\n"
            f"Weekly downloads: {features.get('weekly_downloads', 0):,}\n"
            f"Days since created: {features.get('days_since_created', 0)}\n"
            f"Newest maintainer account age: {features.get('maintainer_min_account_age_days', 0)} days\n"
            f"Script suspicion score: {features.get('script_suspicion_score', 0)}"
            f"{script_section}\n\n"
            f"Top model risk signals (SHAP):\n{top_factors}\n\n"
            f"In exactly 2 sentences: explain what specifically makes this package suspicious "
            f"and what a developer should verify before using it. Be concrete, not generic."
        )

        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return None
