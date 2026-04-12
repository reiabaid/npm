"""Caching layer for SCOPE — stores analysis results with configurable expiry"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.cli.config import ScopeConfig


class ScopeCache:
    """JSON file-based cache for package analysis results with TTL support."""
    
    @staticmethod
    def _get_cache_path(package_name: str) -> Path:
        """Get cache file path for a package."""
        # Hash package name to avoid filesystem issues with special chars
        name_hash = hashlib.md5(package_name.encode()).hexdigest()
        cache_file = ScopeConfig.CACHE_DIR / f"{package_name}_{name_hash}.json"
        return cache_file
    
    @staticmethod
    def _is_expired(cache_data: dict, expiry_hours: int) -> bool:
        """Check if cache entry is older than expiry_hours."""
        if "timestamp" not in cache_data:
            return True
        
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        age = datetime.now() - cache_time
        return age > timedelta(hours=expiry_hours)
    
    @classmethod
    def get(cls, package_name: str) -> Optional[dict]:
        """
        Retrieve cached analysis result if it exists and hasn't expired.
        Returns None if cache miss or expired.
        """
        cache_file = cls._get_cache_path(package_name)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            expiry_hours = ScopeConfig.get("cache_expiry_hours", 1)
            
            if cls._is_expired(cache_data, expiry_hours):
                # Expired — delete and return None
                cache_file.unlink()
                return None
            
            # Still valid — return the result
            return cache_data.get("result")
        
        except (json.JSONDecodeError, IOError, KeyError):
            # Corrupted cache file — silently remove it
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    @classmethod
    def set(cls, package_name: str, result: dict):
        """Cache an analysis result with current timestamp."""
        ScopeConfig._ensure_dirs()
        
        cache_file = cls._get_cache_path(package_name)
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except IOError:
            # Silently fail if we can't write cache (e.g., permission issues)
            pass
    
    @classmethod
    def clear(cls):
        """Clear all cached entries."""
        if ScopeConfig.CACHE_DIR.exists():
            for cache_file in ScopeConfig.CACHE_DIR.glob("*.json"):
                try:
                    cache_file.unlink()
                except:
                    pass
    
    @classmethod
    def clear_package(cls, package_name: str):
        """Clear cache for a specific package."""
        cache_file = cls._get_cache_path(package_name)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
        except:
            pass
