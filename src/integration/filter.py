"""
Cascading filter pipeline module.

Applies a sequence of filters to narrow down the restaurant dataset
based on user preferences. Includes progressive filter relaxation
when too few results are found.
"""

import logging

import pandas as pd
from fuzzywuzzy import process as fuzz_process

from src.data.schema import CleanColumns
from src.input.models import UserPreferences

logger = logging.getLogger("zomato_recommender.integration.filter")

# Minimum acceptable result count before relaxation kicks in
MIN_RESULTS = 5


def apply_filters(
    df: pd.DataFrame,
    prefs: UserPreferences,
    min_results: int = MIN_RESULTS,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply cascading filters to the dataset based on user preferences.

    If the result set is too small (< min_results), filters are
    progressively relaxed until enough results are found.

    Args:
        df: Preprocessed restaurant DataFrame.
        prefs: User preferences to filter by.
        min_results: Minimum number of results before relaxation.

    Returns:
        Tuple of (filtered DataFrame, metadata dict with filter info).
    """
    metadata = {
        "original_count": len(df),
        "filters_applied": [],
        "relaxations": [],
        "final_count": 0,
    }

    # First pass: apply all filters
    filtered = _apply_all_filters(df, prefs, metadata)

    # If enough results, return
    if len(filtered) >= min_results:
        metadata["final_count"] = len(filtered)
        logger.info(
            f"Filtering complete: {metadata['original_count']:,} -> "
            f"{len(filtered):,} restaurants"
        )
        return filtered, metadata

    # Progressive relaxation
    logger.info(
        f"Only {len(filtered)} results found. Starting filter relaxation..."
    )
    filtered = _relax_filters(df, prefs, min_results, metadata)

    metadata["final_count"] = len(filtered)
    logger.info(
        f"After relaxation: {len(filtered):,} restaurants "
        f"(relaxations: {metadata['relaxations']})"
    )

    return filtered, metadata


def _apply_all_filters(
    df: pd.DataFrame,
    prefs: UserPreferences,
    metadata: dict,
) -> pd.DataFrame:
    """Apply all filters in cascade order."""
    result = df.copy()

    # 1. Location filter
    result = _filter_by_location(result, prefs.location)
    if len(result) > 0:
        metadata["filters_applied"].append(f"location={prefs.location}")

    # 2. Budget filter
    if prefs.budget and len(result) > 0:
        budget_result = _filter_by_budget(result, prefs.budget, prefs.party_size)
        if len(budget_result) > 0:
            result = budget_result
            metadata["filters_applied"].append(f"budget={prefs.budget}")

    # 3. Cuisine filter
    if prefs.cuisine and len(result) > 0:
        cuisine_result = _filter_by_cuisine(result, prefs.cuisine)
        if len(cuisine_result) > 0:
            result = cuisine_result
            metadata["filters_applied"].append(f"cuisine={prefs.cuisine}")

    # 4. Rating filter
    if prefs.min_rating > 0 and len(result) > 0:
        rating_result = _filter_by_rating(result, prefs.min_rating)
        if len(rating_result) > 0:
            result = rating_result
            metadata["filters_applied"].append(f"min_rating={prefs.min_rating}")

    return result


def _relax_filters(
    df: pd.DataFrame,
    prefs: UserPreferences,
    min_results: int,
    metadata: dict,
) -> pd.DataFrame:
    """
    Progressively relax filters until min_results are found.

    Relaxation order:
        1. Drop cuisine filter
        2. Expand budget by one tier
        3. Lower min rating by 0.5
        4. Drop location filter (last resort)
    """
    # Start with location only
    result = _filter_by_location(df, prefs.location)

    # Try relaxation steps: prioritize preserving user's cuisine preference
    relaxation_steps = [
        ("rating_lowered", lambda r: _try_lower_rating(r, prefs)),
        ("budget_expanded", lambda r: _try_expanded_budget(r, prefs)),
        ("cuisine_relaxed", lambda r: _try_without_cuisine(r, prefs)),
        ("location_relaxed", lambda _: _try_without_location(df, prefs)),
    ]

    for step_name, step_fn in relaxation_steps:
        candidate = step_fn(result)
        if len(candidate) >= min_results:
            metadata["relaxations"].append(step_name)
            return candidate

    # If steps on location-filtered subset did not yield enough, relax across the full dataset
    candidate_all = _try_without_location(df, prefs)
    if len(candidate_all) >= min_results:
        metadata["relaxations"].append("location_relaxed")
        return candidate_all
    elif len(candidate_all) > 0:
        metadata["relaxations"].append("location_relaxed")
        return candidate_all

    # If nothing works, return whatever location matches we have
    if len(result) == 0:
        # Absolute fallback: just filter by location loosely
        result = _filter_by_location_fuzzy(df, prefs.location)
        if len(result) > 0:
            metadata["relaxations"].append("fuzzy_location")

    # If still completely empty, fallback to the top-rated restaurants in df
    if len(result) == 0 and len(df) > 0:
        metadata["relaxations"].append("top_overall_fallback")
        result = df.sort_values(by=CleanColumns.RATING, ascending=False).head(min_results * 2)

    return result


# ---------------------------------------------------------------------------
# Individual Filters
# ---------------------------------------------------------------------------
def _filter_by_location(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Filter by exact, substring, or city-wide location match (case-insensitive)."""
    col = CleanColumns.LOCATION
    if col not in df.columns or not location:
        return df

    import re
    loc_clean = location.strip().lower()

    # Exact match first
    exact_mask = df[col].astype(str).str.strip().str.lower() == loc_clean
    if exact_mask.any():
        # Check if substring match yields significantly more (e.g. Koramangala -> Koramangala 5th block)
        sub_mask = df[col].astype(str).str.lower().str.contains(r'\b' + re.escape(loc_clean), na=False)
        if sub_mask.sum() > exact_mask.sum():
            return df.loc[sub_mask]
        return df.loc[exact_mask]

    # Substring / word boundary match (e.g. 'HSR' -> 'HSR Layout')
    sub_mask = df[col].astype(str).str.lower().str.contains(r'\b' + re.escape(loc_clean), na=False)
    if sub_mask.any() and loc_clean not in ("bangalore", "bengaluru"):
        return df.loc[sub_mask]

    # City-wide Bangalore check: If searching for Bangalore and there is no single 'Bangalore' row
    # (because all restaurants are in Bangalore neighborhoods), match all restaurants in the city!
    if loc_clean in ("bangalore", "bengaluru", "bangalore (all areas)", "all bangalore", "bangalore city"):
        return df

    return df.head(0)


def _filter_by_location_fuzzy(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Filter by fuzzy location match for relaxed search."""
    col = CleanColumns.LOCATION
    if col not in df.columns:
        return df

    locations = df[col].dropna().unique().tolist()
    match_result = fuzz_process.extractOne(location, locations)
    if match_result and match_result[1] >= 60:
        mask = df[col] == match_result[0]
        return df.loc[mask]

    return df.head(0)  # Empty DataFrame


def _filter_by_budget(df: pd.DataFrame, budget: int | str, party_size: int = 2) -> pd.DataFrame:
    """Filter by numeric budget based on cost for two."""
    col = CleanColumns.COST_FOR_TWO
    if col not in df.columns:
        return df

    # Resolve tier string to number if needed
    if isinstance(budget, str):
        tier_to_budget = {"low": 400, "medium": 1000, "high": 2500}
        resolved = tier_to_budget.get(budget.strip().lower())
        if resolved is not None:
            budget = resolved
        else:
            try:
                budget = int(float(budget.strip().replace("₹", "").replace(",", "")))
            except (ValueError, TypeError):
                return df

    # dataset cost is approx cost for two people
    # normalize the user's budget to a max acceptable "cost for two"
    max_cost_for_two = (budget / max(1, party_size)) * 2
    
    mask = df[col] <= max_cost_for_two
    return df.loc[mask]


def _filter_by_cuisine(df: pd.DataFrame, cuisine: str) -> pd.DataFrame:
    """
    Filter by cuisine — checks if the cuisine appears in the
    restaurant's cuisine list (partial match).
    """
    col = CleanColumns.CUISINES_LIST
    if col not in df.columns:
        # Fallback to string contains on the raw cuisines column
        str_col = CleanColumns.CUISINES
        if str_col in df.columns:
            mask = df[str_col].str.contains(cuisine, case=False, na=False)
            return df.loc[mask]
        return df

    cuisine_lower = cuisine.lower().strip()
    mask = df[col].apply(
        lambda x: any(c.lower().strip() == cuisine_lower for c in x)
        if hasattr(x, "__iter__") and not isinstance(x, str)
        else False
    )
    # If no match in list, also fallback to substring check on raw cuisines text
    if not bool(mask.any()) and CleanColumns.CUISINES in df.columns:
        import re
        mask = df[CleanColumns.CUISINES].astype(str).str.contains(re.escape(cuisine), case=False, na=False)

    return df.loc[mask]


def _filter_by_rating(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """Filter by minimum rating."""
    col = CleanColumns.RATING
    if col not in df.columns:
        return df

    mask = df[col] >= min_rating
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Relaxation helpers
# ---------------------------------------------------------------------------
def _try_without_cuisine(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """Try filtering without the cuisine constraint."""
    result = df
    if prefs.budget:
        budget_result = _filter_by_budget(result, prefs.budget, prefs.party_size)
        if len(budget_result) > 0:
            result = budget_result
    if prefs.min_rating > 0:
        rating_result = _filter_by_rating(result, prefs.min_rating)
        if len(rating_result) > 0:
            result = rating_result
    return result


def _try_expanded_budget(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """Try expanding budget by 50% while preserving cuisine if specified."""
    result = df
    if prefs.cuisine:
        cuisine_result = _filter_by_cuisine(result, prefs.cuisine)
        if len(cuisine_result) > 0:
            result = cuisine_result

    if prefs.budget:
        budget = prefs.budget
        if isinstance(budget, str):
            tier_to_budget = {"low": 400, "medium": 1000, "high": 2500}
            resolved = tier_to_budget.get(budget.strip().lower())
            if resolved is not None:
                budget = resolved
            else:
                try:
                    budget = int(float(budget.strip().replace("₹", "").replace(",", "")))
                except (ValueError, TypeError):
                    budget = None

        if budget is not None:
            col = CleanColumns.COST_FOR_TWO
            if col in result.columns:
                expanded_budget = int(budget * 1.5)
                max_cost_for_two = (expanded_budget / max(1, prefs.party_size)) * 2
                mask = result[col] <= max_cost_for_two
                if mask.any():
                    result = result.loc[mask]

    # Also apply rating filter
    if prefs.min_rating > 0:
        rating_result = _filter_by_rating(result, prefs.min_rating)
        if len(rating_result) > 0:
            result = rating_result

    return result


def _try_lower_rating(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """Try lowering the minimum rating by 0.5 while preserving cuisine if specified."""
    lower_rating = max(0.0, prefs.min_rating - 0.5)
    result = df

    if prefs.cuisine:
        cuisine_result = _filter_by_cuisine(result, prefs.cuisine)
        if len(cuisine_result) > 0:
            result = cuisine_result

    if prefs.budget:
        budget_result = _filter_by_budget(result, prefs.budget, prefs.party_size)
        if len(budget_result) > 0:
            result = budget_result

    rating_result = _filter_by_rating(result, lower_rating)
    if len(rating_result) > 0:
        result = rating_result
    return result


def _try_without_location(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """Last resort: remove location filter entirely."""
    result = df.copy()

    if prefs.budget:
        budget_result = _filter_by_budget(result, prefs.budget, prefs.party_size)
        if len(budget_result) > 0:
            result = budget_result

    if prefs.cuisine:
        cuisine_result = _filter_by_cuisine(result, prefs.cuisine)
        if len(cuisine_result) > 0:
            result = cuisine_result

    if prefs.min_rating > 0:
        rating_result = _filter_by_rating(result, prefs.min_rating)
        if len(rating_result) > 0:
            result = rating_result

    return result
