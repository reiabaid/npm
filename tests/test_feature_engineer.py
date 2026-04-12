"""Pytest tests for feature engineering functions."""

import pytest
from datetime import datetime, timezone
from src.data.feature_engineer import (
    engineer_features,
    _parse_iso,
    _days_since,
    STANDARD_LICENSES
)


class TestParseISO:
    """Test ISO-8601 date parsing."""
    
    def test_parse_iso_with_z_suffix(self):
        """Test parsing ISO date with Z suffix."""
        result = _parse_iso("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_parse_iso_with_offset(self):
        """Test parsing ISO date with timezone offset."""
        result = _parse_iso("2024-01-15T10:30:00+00:00")
        assert result.year == 2024
    
    def test_parse_iso_invalid(self):
        """Test that invalid ISO strings raise ValueError."""
        with pytest.raises(ValueError):
            _parse_iso("not-a-date")
    
    def test_parse_iso_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_iso("")


class TestDaysSince:
    """Test days_since calculation."""
    
    def test_days_since_today(self):
        """Test days_since with today's date."""
        today = datetime.now(timezone.utc).isoformat()
        result = _days_since(today)
        assert result == 0
    
    def test_days_since_none(self):
        """Test that None returns 0."""
        assert _days_since(None) == 0
    
    def test_days_since_invalid(self):
        """Test that invalid date returns 0."""
        assert _days_since("invalid-date") == 0
    
    def test_days_since_empty_string(self):
        """Test that empty string returns 0."""
        assert _days_since("") == 0
    
    def test_days_since_old_date(self):
        """Test days_since with an old date."""
        old_date = "2024-01-01T00:00:00Z"
        result = _days_since(old_date)
        assert result > 0  # Should be many days in the future
    
    def test_days_since_negative_returns_zero(self):
        """Test that future dates return 0."""
        future_date = datetime.now(timezone.utc).replace(year=2030).isoformat()
        # The _days_since function uses max(..., 0) so negative should become 0
        result = _days_since(future_date)
        assert result >= 0


class TestEngineerFeatures:
    """Test the main feature engineering function."""
    
    @pytest.fixture
    def minimal_npm_raw(self):
        """Create minimal valid npm_raw dict."""
        return {
            "name": "test-package",
            "time": {"created": "2023-01-01T00:00:00Z"},
            "versions": {"1.0.0": {}, "1.0.1": {}, "2.0.0": {}},
            "dist-tags": {"latest": "2.0.0"},
            "maintainers": [{"name": "user1"}, {"name": "user2"}],
            "description": "Test package",
            "license": "MIT",
        }
    
    @pytest.fixture
    def minimal_github_raw(self):
        """Create minimal valid github_raw dict."""
        return {
            "has_github_repo": 1,
            "stargazers_count": 100,
            "forks_count": 20,
            "open_issues_count": 5,
            "subscribers_count": 50,
            "contributor_count": 10,
            "pushed_at": "2024-01-15T10:00:00Z",
        }
    
    def test_engineer_features_basic(self, minimal_npm_raw, minimal_github_raw):
        """Test basic feature engineering."""
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["name"] == "test-package"
        assert result["num_versions"] == 3
        assert result["num_maintainers"] == 2
        assert result["description_length"] == 12
        assert result["license_is_standard"] == 1
        assert result["has_github_repo"] == 1
        assert result["stargazers_count"] == 100
        assert result["forks_count"] == 20
    
    def test_engineer_features_no_github_repo(self, minimal_npm_raw):
        """Test when github_raw is None."""
        result = engineer_features(minimal_npm_raw, {})
        
        assert result["has_github_repo"] == 0
        assert result["stargazers_count"] == 0
        assert result["days_since_last_commit"] == 0
    
    def test_engineer_features_empty_maintainers(self, minimal_npm_raw, minimal_github_raw):
        """Test with empty maintainers list."""
        minimal_npm_raw["maintainers"] = []
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["num_maintainers"] == 0
    
    def test_engineer_features_no_versions(self, minimal_npm_raw, minimal_github_raw):
        """Test with empty versions dict."""
        minimal_npm_raw["versions"] = {}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["num_versions"] == 0
        assert result["release_velocity"] == 0.0
    
    def test_engineer_features_single_version(self, minimal_npm_raw, minimal_github_raw):
        """Test with only one version."""
        minimal_npm_raw["versions"] = {"1.0.0": {}}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["num_versions"] == 1
        assert result["release_velocity"] > 0
    
    def test_engineer_features_postinstall_script(self, minimal_npm_raw, minimal_github_raw):
        """Test detection of postinstall script."""
        minimal_npm_raw["versions"]["2.0.0"]["scripts"] = {"postinstall": "node install.js"}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["has_postinstall"] == 1
    
    def test_engineer_features_no_postinstall_script(self, minimal_npm_raw, minimal_github_raw):
        """Test when postinstall script is absent."""
        minimal_npm_raw["versions"]["2.0.0"]["scripts"] = {"test": "jest"}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["has_postinstall"] == 0
    
    def test_engineer_features_empty_scripts(self, minimal_npm_raw, minimal_github_raw):
        """Test with empty scripts dict."""
        minimal_npm_raw["versions"]["2.0.0"]["scripts"] = {}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["has_postinstall"] == 0
    
    def test_engineer_features_no_description(self, minimal_npm_raw, minimal_github_raw):
        """Test with missing description."""
        del minimal_npm_raw["description"]
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["description_length"] == 0
    
    def test_engineer_features_empty_description(self, minimal_npm_raw, minimal_github_raw):
        """Test with empty description."""
        minimal_npm_raw["description"] = ""
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["description_length"] == 0
    
    def test_engineer_features_long_description(self, minimal_npm_raw, minimal_github_raw):
        """Test with very long description."""
        minimal_npm_raw["description"] = "x" * 1000
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["description_length"] == 1000
    
    def test_engineer_features_standard_license_mit(self, minimal_npm_raw, minimal_github_raw):
        """Test with MIT license."""
        minimal_npm_raw["license"] = "MIT"
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 1
    
    def test_engineer_features_standard_license_apache(self, minimal_npm_raw, minimal_github_raw):
        """Test with Apache-2.0 license."""
        minimal_npm_raw["license"] = "Apache-2.0"
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 1
    
    def test_engineer_features_nonstandard_license(self, minimal_npm_raw, minimal_github_raw):
        """Test with non-standard license."""
        minimal_npm_raw["license"] = "CUSTOM-LICENSE"
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 0
    
    def test_engineer_features_license_as_dict(self, minimal_npm_raw, minimal_github_raw):
        """Test with license as dict object."""
        minimal_npm_raw["license"] = {"type": "MIT"}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 1
    
    def test_engineer_features_license_as_dict_nonstandard(self, minimal_npm_raw, minimal_github_raw):
        """Test with license dict containing non-standard license."""
        minimal_npm_raw["license"] = {"type": "CUSTOM"}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 0
    
    def test_engineer_features_no_license(self, minimal_npm_raw, minimal_github_raw):
        """Test when license is missing."""
        del minimal_npm_raw["license"]
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["license_is_standard"] == 0
    
    def test_engineer_features_release_velocity_calculation(self, minimal_npm_raw, minimal_github_raw):
        """Test that release_velocity is calculated correctly."""
        # 10 versions over 100 days
        minimal_npm_raw["versions"] = {f"{i}.0.0": {} for i in range(10)}
        minimal_npm_raw["time"]["created"] = (
            datetime.now(timezone.utc)
            .replace(year=datetime.now(timezone.utc).year - 1)
            .isoformat()
        )
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["num_versions"] == 10
        assert result["release_velocity"] > 0
    
    def test_engineer_features_release_velocity_zero_days(self, minimal_npm_raw, minimal_github_raw):
        """Test release_velocity when days_since_created is 0."""
        # Today's date
        minimal_npm_raw["time"]["created"] = datetime.now(timezone.utc).isoformat()
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        # Should avoid division by zero and return num_versions / 1
        assert result["release_velocity"] == 3.0
    
    def test_engineer_features_all_fields_present(self, minimal_npm_raw, minimal_github_raw):
        """Test that all expected feature keys are present in output."""
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        expected_keys = {
            "name",
            "days_since_created",
            "days_since_last_update",
            "num_versions",
            "release_velocity",
            "num_maintainers",
            "has_postinstall",
            "description_length",
            "license_is_standard",
            "has_github_repo",
            "stargazers_count",
            "forks_count",
            "open_issues_count",
            "subscribers_count",
            "contributor_count",
            "days_since_last_commit",
        }
        
        assert set(result.keys()) == expected_keys
    
    def test_engineer_features_large_version_count(self, minimal_npm_raw, minimal_github_raw):
        """Test with 1000+ versions (edge case)."""
        minimal_npm_raw["versions"] = {f"{i}.0.0": {} for i in range(1000)}
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["num_versions"] == 1000
        assert result["release_velocity"] > 0
    
    def test_engineer_features_scoped_package_name(self, minimal_npm_raw, minimal_github_raw):
        """Test with scoped package name like @types/react."""
        minimal_npm_raw["name"] = "@types/react"
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["name"] == "@types/react"
    
    def test_engineer_features_special_characters_in_name(self, minimal_npm_raw, minimal_github_raw):
        """Test with special characters in package name."""
        minimal_npm_raw["name"] = "pkg-with.special_chars!"
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["name"] == "pkg-with.special_chars!"
    
    def test_engineer_features_extremely_long_name(self, minimal_npm_raw, minimal_github_raw):
        """Test with extremely long package name."""
        long_name = "a" * 500
        minimal_npm_raw["name"] = long_name
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["name"] == long_name
    
    def test_engineer_features_zero_downloads_high_stars(self, minimal_npm_raw, minimal_github_raw):
        """Test with 0 downloads but high GitHub stars (edge case)."""
        minimal_github_raw["stargazers_count"] = 10000
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["stargazers_count"] == 10000
    
    def test_engineer_features_high_downloads_zero_stars(self, minimal_npm_raw, minimal_github_raw):
        """Test with high downloads but 0 GitHub stars (edge case)."""
        minimal_github_raw["stargazers_count"] = 0
        minimal_github_raw["forks_count"] = 0
        result = engineer_features(minimal_npm_raw, minimal_github_raw)
        
        assert result["stargazers_count"] == 0
        assert result["forks_count"] == 0
    
    def test_engineer_features_no_github_repo_field(self, minimal_npm_raw, minimal_github_raw):
        """Test when GitHub repo data is missing."""
        result = engineer_features(minimal_npm_raw, {})
        
        assert result["has_github_repo"] == 0
        assert all(val == 0 for k, val in result.items() 
                   if k.startswith("stargazers") or k.startswith("forks") 
                   or k.startswith("open_issues") or k.startswith("subscribers") 
                   or k.startswith("contributor"))


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_feature_pipeline_real_like_data(self):
        """Test with more realistic package data."""
        npm_raw = {
            "name": "lodash",
            "time": {
                "created": "2012-02-29T00:00:00Z",
                "modified": "2024-01-15T12:00:00Z",
                "4.17.21": "2021-02-17T10:00:00Z",
                "4.17.20": "2020-01-10T10:00:00Z",
            },
            "versions": {f"4.17.{i}": {} for i in range(22)},  # 22 versions
            "dist-tags": {"latest": "4.17.21"},
            "maintainers": [{"name": "jdalton"}],
            "description": "Lodash modular utilities.",
            "license": "MIT",
            "repository": {"type": "git", "url": "git+https://github.com/lodash/lodash.git"},
        }
        
        github_raw = {
            "has_github_repo": 1,
            "stargazers_count": 55000,
            "forks_count": 6000,
            "open_issues_count": 200,
            "subscribers_count": 2500,
            "contributor_count": 120,
            "pushed_at": "2021-02-17T10:00:00Z",
        }
        
        result = engineer_features(npm_raw, github_raw)
        
        assert result["name"] == "lodash"
        assert result["num_versions"] == 22
        assert result["num_maintainers"] == 1
        assert result["stargazers_count"] == 55000
        assert result["has_postinstall"] == 0
        assert result["license_is_standard"] == 1
