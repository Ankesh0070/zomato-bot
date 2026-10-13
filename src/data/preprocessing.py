"""
Data preprocessing module — clean, normalize, and enrich the Zomato dataset.

Transforms the raw Hugging Face dataset into a clean, analysis-ready
DataFrame with normalized text, parsed numbers, and derived columns.
"""

import logging
import re

import numpy as np
import pandas as pd

from src.data.schema import (
    REQUIRED_COLUMNS,
    CleanColumns,
    RawColumns,
    get_budget_tier,
)

logger = logging.getLogger("zomato_recommender.data.preprocessing")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the raw Zomato dataset.

    Steps:
        1. Rename columns to standardized names
        2. Parse and clean ratings
        3. Parse and clean costs
        4. Normalize text fields (location, cuisines)
        5. Derive new columns (budget_tier, cuisines_list)
        6. Remove duplicates
        7. Drop rows with critical missing values
        8. Reset index

    Args:
        df: Raw DataFrame from ingestion.

    Returns:
        Clean, preprocessed DataFrame ready for filtering.
    """
    logger.info(f"Starting preprocessing on {len(df):,} rows...")
    initial_count = len(df)

    # Work on a copy to avoid mutating the original
    df = df.copy()

    # Step 1: Rename columns
    df = _rename_columns(df)

    # Step 2: Parse ratings
    df = _parse_ratings(df)

    # Step 3: Parse costs
    df = _parse_costs(df)

    # Step 4: Parse votes
    df = _parse_votes(df)

    # Step 5: Normalize text fields
    df = _normalize_text(df)

    # Step 5: Derive new columns
    df = _derive_columns(df)

    # Step 6: Parse boolean fields
    df = _parse_booleans(df)

    # Step 7: Remove duplicates
    df = _remove_duplicates(df)

    # Step 8: Drop rows with critical nulls
    df = _drop_critical_nulls(df)

    # Step 9: Reset index
    df = df.reset_index(drop=True)

    final_count = len(df)
    dropped = initial_count - final_count
    logger.info(
        f"Preprocessing complete: {final_count:,} rows "
        f"({dropped:,} rows dropped, {dropped / max(initial_count, 1) * 100:.1f}%)"
    )

    return df


# ---------------------------------------------------------------------------
# Step 1: Rename columns
# ---------------------------------------------------------------------------
def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column names to standardized names."""
    rename_map = {
        RawColumns.RESTAURANT_NAME: CleanColumns.RESTAURANT_NAME,
        RawColumns.ONLINE_ORDER: CleanColumns.ONLINE_ORDER,
        RawColumns.BOOK_TABLE: CleanColumns.BOOK_TABLE,
        RawColumns.RATING: CleanColumns.RATING,
        RawColumns.VOTES: CleanColumns.VOTES,
        RawColumns.LOCATION: CleanColumns.LOCATION,
        RawColumns.REST_TYPE: CleanColumns.RESTAURANT_TYPE,
        RawColumns.CUISINES: CleanColumns.CUISINES,
        RawColumns.COST_FOR_TWO: CleanColumns.COST_FOR_TWO,
        RawColumns.LISTED_IN_TYPE: CleanColumns.LISTED_IN_TYPE,
        RawColumns.LISTED_IN_CITY: CleanColumns.LISTED_IN_CITY,
    }

    # Only rename columns that exist in the DataFrame
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}

    if existing_renames:
        df = df.rename(columns=existing_renames)
        logger.debug(f"Renamed {len(existing_renames)} columns.")

    # Check for required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning(f"Missing required columns after rename: {missing}")

    return df


# ---------------------------------------------------------------------------
# Step 2: Parse ratings
# ---------------------------------------------------------------------------
def _parse_rating_value(value) -> float | None:
    """
    Parse a single rating value to float.

    Handles formats: '4.1/5', '4.1', 'NEW', '-', '', None, NaN
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    value_str = str(value).strip()

    # Handle known non-numeric values
    if value_str.lower() in ("new", "-", "", "none", "nan", "--"):
        return None

    # Remove '/5' suffix
    value_str = re.sub(r"\s*/\s*5\s*$", "", value_str)

    try:
        rating = float(value_str)
        # Clamp to valid range
        if rating < 0:
            return None
        if rating > 5:
            return 5.0
        return round(rating, 1)
    except (ValueError, TypeError):
        return None


def _parse_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the rating column to clean floats."""
    col = CleanColumns.RATING
    if col not in df.columns:
        logger.warning(f"Rating column '{col}' not found.")
        return df

    before_nulls = df[col].isnull().sum()
    df[col] = df[col].apply(_parse_rating_value)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    after_nulls = df[col].isnull().sum()

    new_nulls = after_nulls - before_nulls
    if new_nulls > 0:
        logger.info(f"Ratings: {new_nulls:,} unparseable values set to NaN.")

    return df


# ---------------------------------------------------------------------------
# Step 3: Parse costs
# ---------------------------------------------------------------------------
def _parse_cost_value(value) -> int | None:
    """
    Parse a single cost value to int.

    Handles formats: '800', '1,200', '₹800', '', 'N/A', None, NaN
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    value_str = str(value).strip()

    # Handle known non-numeric values
    if value_str.lower() in ("", "none", "nan", "n/a", "-"):
        return None

    # Remove currency symbols, commas, spaces
    value_str = re.sub(r"[₹,\s]", "", value_str)

    try:
        cost = int(float(value_str))
        if cost < 0:
            return None
        return cost
    except (ValueError, TypeError):
        return None


def _parse_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the cost column to clean integers."""
    col = CleanColumns.COST_FOR_TWO
    if col not in df.columns:
        logger.warning(f"Cost column '{col}' not found.")
        return df

    before_nulls = df[col].isnull().sum()
    df[col] = df[col].apply(_parse_cost_value)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    after_nulls = df[col].isnull().sum()

    new_nulls = after_nulls - before_nulls
    if new_nulls > 0:
        logger.info(f"Costs: {new_nulls:,} unparseable values set to NaN.")

    return df


# ---------------------------------------------------------------------------
# Step 4: Parse votes
# ---------------------------------------------------------------------------
def _parse_vote_value(value) -> int:
    """
    Parse a single vote count to a clean non-negative integer.

    Handles formats: 100, '100', '1,200', '0', None, NaN, etc.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0

    value_str = str(value).strip()
    value_str = re.sub(r"[,\s]", "", value_str)

    try:
        vote = int(float(value_str))
        return max(0, vote)
    except (ValueError, TypeError):
        return 0


def _parse_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the vote column to clean integers."""
    col = CleanColumns.VOTES
    if col not in df.columns:
        return df

    df[col] = df[col].apply(_parse_vote_value)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df



# ---------------------------------------------------------------------------
# Step 4: Normalize text fields
# ---------------------------------------------------------------------------
def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns: strip whitespace, title case names."""

    # Restaurant name — title case, strip
    if CleanColumns.RESTAURANT_NAME in df.columns:
        df[CleanColumns.RESTAURANT_NAME] = (
            df[CleanColumns.RESTAURANT_NAME]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Location — title case, strip
    if CleanColumns.LOCATION in df.columns:
        df[CleanColumns.LOCATION] = (
            df[CleanColumns.LOCATION]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Cuisines — strip (keep original casing for display)
    if CleanColumns.CUISINES in df.columns:
        df[CleanColumns.CUISINES] = (
            df[CleanColumns.CUISINES]
            .astype(str)
            .str.strip()
        )

    # Restaurant type — strip
    if CleanColumns.RESTAURANT_TYPE in df.columns:
        df[CleanColumns.RESTAURANT_TYPE] = (
            df[CleanColumns.RESTAURANT_TYPE]
            .astype(str)
            .str.strip()
        )

    logger.debug("Text fields normalized.")
    return df


# ---------------------------------------------------------------------------
# Step 5: Derive new columns
# ---------------------------------------------------------------------------
def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns: budget_tier, cuisines_list."""

    # Budget tier
    if CleanColumns.COST_FOR_TWO in df.columns:
        df[CleanColumns.BUDGET_TIER] = df[CleanColumns.COST_FOR_TWO].apply(
            get_budget_tier
        )
        tier_counts = df[CleanColumns.BUDGET_TIER].value_counts().to_dict()
        logger.debug(f"Budget tier distribution: {tier_counts}")

    # Cuisines list (split comma-separated string into list)
    if CleanColumns.CUISINES in df.columns:
        df[CleanColumns.CUISINES_LIST] = df[CleanColumns.CUISINES].apply(
            _split_cuisines
        )

    return df


def _split_cuisines(cuisines_str) -> list[str]:
    """
    Split a comma-separated cuisine string into a normalized list.

    Examples:
        'Italian, Chinese'  → ['Italian', 'Chinese']
        'North Indian'      → ['North Indian']
        ''                  → []
        None                → []
    """
    if not cuisines_str or str(cuisines_str).lower() in ("nan", "none", ""):
        return []

    return [c.strip() for c in str(cuisines_str).split(",") if c.strip()]


# ---------------------------------------------------------------------------
# Step 6: Parse boolean fields
# ---------------------------------------------------------------------------
def _parse_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No string columns to boolean."""

    for col in [CleanColumns.ONLINE_ORDER, CleanColumns.BOOK_TABLE]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": True, "no": False})
                .fillna(False)
            )

    logger.debug("Boolean fields parsed.")
    return df


# ---------------------------------------------------------------------------
# Step 7: Remove duplicates
# ---------------------------------------------------------------------------
def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate restaurants.

    Deduplicates by (restaurant_name, location), keeping the entry
    with the highest vote count.
    """
    name_col = CleanColumns.RESTAURANT_NAME
    loc_col = CleanColumns.LOCATION
    votes_col = CleanColumns.VOTES

    if name_col not in df.columns or loc_col not in df.columns:
        return df

    before = len(df)

    # Sort by votes (descending) so the first occurrence has the most votes
    if votes_col in df.columns:
        df = df.sort_values(votes_col, ascending=False, na_position="last")

    df = df.drop_duplicates(subset=[name_col, loc_col], keep="first")

    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate restaurants.")

    return df


# ---------------------------------------------------------------------------
# Step 8: Drop critical nulls
# ---------------------------------------------------------------------------
def _drop_critical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where critical columns (name, location, rating) are null."""
    critical = [col for col in REQUIRED_COLUMNS if col in df.columns]

    if not critical:
        return df

    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)

    if dropped > 0:
        logger.info(f"Dropped {dropped:,} rows with null critical fields: {critical}")

    return df


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------
def get_unique_locations(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique locations in the dataset."""
    col = CleanColumns.LOCATION
    if col not in df.columns:
        return []
    locs = df[col].dropna().unique().tolist()
    return sorted(loc for loc in locs if loc and str(loc).strip() and str(loc).lower() != "nan")


def get_unique_cuisines(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique individual cuisines across all restaurants."""
    col = CleanColumns.CUISINES_LIST
    if col not in df.columns:
        return []

    all_cuisines = set()
    for cuisine_list in df[col].dropna():
        if hasattr(cuisine_list, "__iter__") and not isinstance(cuisine_list, str):
            for c in cuisine_list:
                if c and str(c).strip() and str(c).lower() != "nan":
                    all_cuisines.add(str(c).strip())

    return sorted(all_cuisines)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the preprocessed dataset.

    Returns:
        Dictionary with key statistics.
    """
    summary = {
        "total_restaurants": len(df),
        "unique_locations": len(get_unique_locations(df)),
        "unique_cuisines": len(get_unique_cuisines(df)),
        "rating_range": (
            round(df[CleanColumns.RATING].min(), 1) if CleanColumns.RATING in df.columns else None,
            round(df[CleanColumns.RATING].max(), 1) if CleanColumns.RATING in df.columns else None,
        ),
        "cost_range": (
            int(df[CleanColumns.COST_FOR_TWO].min()) if CleanColumns.COST_FOR_TWO in df.columns else None,
            int(df[CleanColumns.COST_FOR_TWO].max()) if CleanColumns.COST_FOR_TWO in df.columns else None,
        ),
    }

    if CleanColumns.BUDGET_TIER in df.columns:
        summary["budget_distribution"] = df[CleanColumns.BUDGET_TIER].value_counts().to_dict()

    return summary


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.config import setup_logging
    from src.data.ingestion import download_dataset

    setup_logging()

    print("Loading dataset...")
    raw_df = download_dataset()
    print(f"Raw: {len(raw_df):,} rows × {len(raw_df.columns)} columns")

    print("\nPreprocessing...")
    clean_df = preprocess(raw_df)
    print(f"Clean: {len(clean_df):,} rows × {len(clean_df.columns)} columns")

    print(f"\nColumns: {list(clean_df.columns)}")

    summary = get_dataset_summary(clean_df)
    print("\n📊 Dataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n📍 Sample locations: {get_unique_locations(clean_df)[:10]}")
    print(f"🍕 Sample cuisines: {get_unique_cuisines(clean_df)[:10]}")

    print("\nFirst 3 rows:")
    print(clean_df.head(3).to_string())
