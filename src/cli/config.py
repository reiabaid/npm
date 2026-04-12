"""Configuration management for SCOPE — handles ~/.scope/config.json"""

import json
import os
from pathlib import Path
from typing import Optional


class ScopeConfig:
    """Manage SCOPE configuration stored in ~/.scope/config.json"""
    
    CONFIG_DIR = Path.home() / ".scope"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    CACHE_DIR = CONFIG_DIR / "cache"
    
    DEFAULT_CONFIG = {
        "github_token": "",
        "npm_timeout": 30,
        "github_timeout": 30,
        "cache_expiry_hours": 1,
    }
    
    @classmethod
    def _ensure_dirs(cls):
        """Create ~/.scope and ~/.scope/cache directories if they don't exist."""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls) -> dict:
        """Load config from file or return defaults if file doesn't exist."""
        cls._ensure_dirs()
        
        if cls.CONFIG_FILE.exists():
            try:
                with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return cls.DEFAULT_CONFIG.copy()
        
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save(cls, config: dict):
        """Save config to ~/.scope/config.json"""
        cls._ensure_dirs()
        
        with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def get(cls, key: str, default=None):
        """Get a single config value."""
        config = cls.load()
        return config.get(key, default)
    
    @classmethod
    def set(cls, key: str, value):
        """Set a single config value."""
        config = cls.load()
        config[key] = value
        cls.save(config)
    
    @classmethod
    def set_github_token(cls, token: str):
        """Set GitHub API token for faster rate limits."""
        cls.set("github_token", token)
    
    @classmethod
    def get_github_token(cls) -> Optional[str]:
        """Get GitHub API token."""
        token = cls.get("github_token", "").strip()
        return token if token else None
