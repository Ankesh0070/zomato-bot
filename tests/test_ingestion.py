"""
Unit tests for the data ingestion and preprocessing modules.

Tests cover:
- Schema constants and budget tier logic
- Rating parsing edge cases
- Cost parsing edge cases
- Cuisine splitting
- Preprocessing pipeline end-to-end
- Duplicate removal
- Critical null dropping
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import (
    CleanColumns,
    RawColumns,
    BUDGET_TIERS,
    get_budget_tier,
    Restaurant,
)
from src.data.preprocessing import (
    preprocess,
    _parse_rating_value,
    _parse_cost_value,
    _split_cuisines,
    get_unique_locations,
    get_unique_cuisines,
)


# ===================================================================
# Schema Tests
# ===================================================================

class TestBudgetTier:
    """Tests for budget tier classification."""

    def test_low_budget(self):
        assert get_budget_tier(200) == "low"

    def test_low_budget_boundary(self):
        assert get_budget_tier(400) == "low"

    def test_medium_budget(self):
        assert get_budget_tier(600) == "medium"

    def test_medium_budget_boundary(self):
        assert get_budget_tier(1000) == "medium"

    def test_high_budget(self):
        assert get_budget_tier(1500) == "high"

    def test_high_budget_boundary(self):
        assert get_budget_tier(1001) == "high"

    def test_zero_cost(self):
        assert get_budget_tier(0) == "low"

    def test_none_cost(self):
        assert get_budget_tier(None) is None

    def test_negative_cost(self):
        assert get_budget_tier(-100) is None

    def test_very_high_cost(self):
        assert get_budget_tier(10000) == "high"


class TestRestaurantModel:
    """Tests for the Restaurant dataclass."""

    def test_create_restaurant(self):
        r = Restaurant(
            restaurant_name="Test Restaurant",
            location="Delhi",
            rating=4.5,
            cost_for_two=800,
        )
        assert r.restaurant_name == "Test Restaurant"
        assert r.rating == 4.5

    def test_matches_cuisine_positive(self):
        r = Restaurant(
            restaurant_name="Test",
            location="Delhi",
            rating=4.0,
            cost_for_two=500,
            cuisines_list=["Italian", "Chinese"],
        )
        assert r.matches_cuisine("Italian") is True
        assert r.matches_cuisine("italian") is True  # Case insensitive

    def test_matches_cuisine_negative(self):
        r = Restaurant(
            restaurant_name="Test",
            location="Delhi",
            rating=4.0,
            cost_for_two=500,
            cuisines_list=["Italian"],
        )
        assert r.matches_cuisine("Chinese") is False

    def test_matches_cuisine_empty(self):
        r = Restaurant(
            restaurant_name="Test",
            location="Delhi",
            rating=4.0,
            cost_for_two=500,
            cuisines_list=None,
        )
        assert r.matches_cuisine("Italian") is False

    def test_to_display_dict(self):
        r = Restaurant(
            restaurant_name="La Piazza",
            location="Delhi",
            rating=4.5,
            cost_for_two=800,
            cuisines="Italian, Continental",
            votes=100,
        )
        d = r.to_display_dict()
        assert d["Restaurant Name"] == "La Piazza"
        assert d["Cost for Two"] == "₹800"
        assert d["Rating"] == "4.5/5"


# ===================================================================
# Rating Parsing Tests
# ===================================================================

class TestRatingParsing:
    """Tests for rating value parsing edge cases."""

    def test_normal_rating(self):
        assert _parse_rating_value("4.1") == 4.1

    def test_rating_with_slash_5(self):
        assert _parse_rating_value("4.1/5") == 4.1

    def test_rating_with_spaced_slash(self):
        assert _parse_rating_value("4.1 / 5") == 4.1

    def test_rating_new(self):
        assert _parse_rating_value("NEW") is None

    def test_rating_dash(self):
        assert _parse_rating_value("-") is None

    def test_rating_empty_string(self):
        assert _parse_rating_value("") is None

    def test_rating_none(self):
        assert _parse_rating_value(None) is None

    def test_rating_nan(self):
        assert _parse_rating_value(float("nan")) is None

    def test_rating_with_whitespace(self):
        assert _parse_rating_value("  3.5  ") == 3.5

    def test_rating_above_5_clamped(self):
        assert _parse_rating_value("6.0") == 5.0

    def test_rating_negative(self):
        assert _parse_rating_value("-1") is None

    def test_rating_zero(self):
        assert _parse_rating_value("0") == 0.0

    def test_rating_integer(self):
        assert _parse_rating_value("4") == 4.0


# ===================================================================
# Cost Parsing Tests
# ===================================================================

class TestCostParsing:
    """Tests for cost value parsing edge cases."""

    def test_normal_cost(self):
        assert _parse_cost_value("800") == 800

    def test_cost_with_comma(self):
        assert _parse_cost_value("1,200") == 1200

    def test_cost_with_currency(self):
        assert _parse_cost_value("₹800") == 800

    def test_cost_empty(self):
        assert _parse_cost_value("") is None

    def test_cost_na(self):
        assert _parse_cost_value("N/A") is None

    def test_cost_none(self):
        assert _parse_cost_value(None) is None

    def test_cost_nan(self):
        assert _parse_cost_value(float("nan")) is None

    def test_cost_negative(self):
        assert _parse_cost_value("-100") is None

    def test_cost_with_spaces(self):
        assert _parse_cost_value("  800  ") == 800

    def test_cost_float_string(self):
        assert _parse_cost_value("799.50") == 799


# ===================================================================
# Cuisine Splitting Tests
# ===================================================================

class TestCuisineSplitting:
    """Tests for cuisine string splitting."""

    def test_multiple_cuisines(self):
        assert _split_cuisines("Italian, Chinese") == ["Italian", "Chinese"]

    def test_single_cuisine(self):
        assert _split_cuisines("North Indian") == ["North Indian"]

    def test_empty_string(self):
        assert _split_cuisines("") == []

    def test_none_value(self):
        assert _split_cuisines(None) == []

    def test_nan_string(self):
        assert _split_cuisines("nan") == []

    def test_extra_whitespace(self):
        result = _split_cuisines("  Italian ,  Chinese  , Thai  ")
        assert result == ["Italian", "Chinese", "Thai"]

    def test_trailing_comma(self):
        result = _split_cuisines("Italian, Chinese,")
        assert result == ["Italian", "Chinese"]


# ===================================================================
# Full Preprocessing Pipeline Tests
# ===================================================================

def _make_raw_dataframe(n: int = 5) -> pd.DataFrame:
    """Create a sample raw DataFrame mimicking the HF dataset structure."""
    return pd.DataFrame({
        RawColumns.RESTAURANT_NAME: [f"Restaurant {i}" for i in range(n)],
        RawColumns.ONLINE_ORDER: ["Yes", "No", "Yes", "Yes", "No"][:n],
        RawColumns.BOOK_TABLE: ["Yes", "No", "No", "Yes", "No"][:n],
        RawColumns.RATING: ["4.1/5", "3.5/5", "NEW", "4.8/5", "2.0/5"][:n],
        RawColumns.VOTES: [500, 200, 0, 1000, 50][:n],
        RawColumns.LOCATION: ["Delhi", "Bangalore", "Mumbai", "Delhi", "Chennai"][:n],
        RawColumns.REST_TYPE: ["Casual Dining", "Cafe", "QSR", "Fine Dining", "Cafe"][:n],
        RawColumns.CUISINES: [
            "Italian, Chinese",
            "North Indian",
            "Chinese, Thai",
            "Italian, Continental",
            "South Indian",
        ][:n],
        RawColumns.COST_FOR_TWO: ["800", "300", "400", "1,500", "200"][:n],
        RawColumns.LISTED_IN_TYPE: ["Delivery", "Dine-out", "Delivery", "Dine-out", "Delivery"][:n],
        RawColumns.LISTED_IN_CITY: ["Delhi", "Bangalore", "Mumbai", "Delhi", "Chennai"][:n],
    })


class TestPreprocessPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def test_pipeline_runs_without_error(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        assert len(clean) > 0

    def test_columns_renamed(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        assert CleanColumns.RESTAURANT_NAME in clean.columns
        assert CleanColumns.RATING in clean.columns
        assert CleanColumns.COST_FOR_TWO in clean.columns

    def test_ratings_are_float(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        # Filter non-null ratings
        valid_ratings = clean[CleanColumns.RATING].dropna()
        assert all(isinstance(r, float) for r in valid_ratings)
        assert all(0 <= r <= 5 for r in valid_ratings)

    def test_costs_are_numeric(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        valid_costs = clean[CleanColumns.COST_FOR_TWO].dropna()
        assert all(c >= 0 for c in valid_costs)

    def test_budget_tier_derived(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        assert CleanColumns.BUDGET_TIER in clean.columns
        valid_tiers = clean[CleanColumns.BUDGET_TIER].dropna().unique()
        for tier in valid_tiers:
            assert tier in ("low", "medium", "high")

    def test_cuisines_list_derived(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        assert CleanColumns.CUISINES_LIST in clean.columns
        first_row = clean.iloc[0]
        assert isinstance(first_row[CleanColumns.CUISINES_LIST], list)

    def test_booleans_parsed(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        if CleanColumns.ONLINE_ORDER in clean.columns:
            assert clean[CleanColumns.ONLINE_ORDER].dtype == bool

    def test_new_rating_becomes_nan(self):
        """Restaurant with 'NEW' rating should have NaN and be dropped (critical null)."""
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        # "Restaurant 2" had rating "NEW" → NaN → should be dropped
        names = clean[CleanColumns.RESTAURANT_NAME].str.lower().tolist()
        assert "restaurant 2" not in names

    def test_duplicates_removed(self):
        """Duplicate restaurants by (name, location) should be deduplicated."""
        raw = _make_raw_dataframe()
        # Add a duplicate of the first restaurant
        dup_row = raw.iloc[0:1].copy()
        raw = pd.concat([raw, dup_row], ignore_index=True)
        clean = preprocess(raw)
        # Should have fewer rows than raw (minus the duplicate and the NEW row)
        assert len(clean) < len(raw)

    def test_locations_are_title_case(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        for loc in clean[CleanColumns.LOCATION]:
            assert loc == loc.strip()
            assert loc[0].isupper()


class TestUtilities:
    """Tests for utility functions."""

    def test_get_unique_locations(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        locations = get_unique_locations(clean)
        assert isinstance(locations, list)
        assert len(locations) > 0
        # Should be sorted
        assert locations == sorted(locations)

    def test_get_unique_cuisines(self):
        raw = _make_raw_dataframe()
        clean = preprocess(raw)
        cuisines = get_unique_cuisines(clean)
        assert isinstance(cuisines, list)
        assert len(cuisines) > 0
        # Should be sorted
        assert cuisines == sorted(cuisines)


# ===================================================================
# Run tests
# ===================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
