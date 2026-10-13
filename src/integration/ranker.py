"""
Weighted ranking and top-K selection module.

Ranks filtered restaurants using a composite score based on
rating, votes, and cost proximity to user preference.
Selects the top-K candidates for the LLM prompt.
"""

import logging
import math

import pandas as pd

from src.config import AppConfig
from src.data.schema import CleanColumns

from src.input.models import UserPreferences

logger = logging.getLogger("zomato_recommender.integration.ranker")


def rank_and_select(
    df: pd.DataFrame,
    top_k: int | None = None,
    prefs: UserPreferences | None = None,
) -> pd.DataFrame:
    """
    Rank restaurants by composite score and return the top-K.

    The composite score balances:
        - Rating (primary signal)
        - Popularity (log of votes — diminishing returns)
        - User preference affinity (cuisine match, dining tags, options)

    Args:
        df: Filtered restaurant DataFrame.
        top_k: Number of top candidates to return. Defaults to AppConfig.TOP_K_CANDIDATES.
        prefs: Optional user preferences to calculate personalized bonus.

    Returns:
        Top-K ranked DataFrame sorted by score (descending).
    """
    if top_k is None:
        top_k = AppConfig.TOP_K_CANDIDATES

    if df.empty:
        logger.warning("Empty DataFrame passed to ranker.")
        return df

    df = df.copy()

    # Calculate composite score with preferences affinity
    df["_score"] = df.apply(lambda row: _compute_score(row, prefs), axis=1)

    # Sort by score (descending), then by name (ascending) for determinism
    df = df.sort_values(
        by=["_score", CleanColumns.RESTAURANT_NAME],
        ascending=[False, True],
    )

    # Select top-K
    result = df.head(top_k)

    # Add rank column
    result = result.copy()
    result["_rank"] = range(1, len(result) + 1)

    logger.info(
        f"Ranked {len(df):,} restaurants. "
        f"Selected top {len(result)} (score range: "
        f"{result['_score'].max():.2f} – {result['_score'].min():.2f})"
    )

    return result


def _compute_score(row: pd.Series, prefs: UserPreferences | None = None) -> float:
    """
    Compute the composite ranking score for a single restaurant.

    Returns:
        Float score — higher is better.
    """
    # Get rating (default 0 if missing)
    rating = row.get(CleanColumns.RATING, 0)
    if pd.isna(rating):
        rating = 0.0

    # Get votes (default 0 if missing)
    votes = row.get(CleanColumns.VOTES, 0)
    if pd.isna(votes):
        votes = 0

    # Ensure non-negative
    rating = max(0.0, float(rating))
    votes = max(0, int(votes))

    # Composite score: rating * log2(votes + 2)
    popularity_factor = math.log2(votes + 2)
    score = rating * popularity_factor

    # Personalization bonus based on user preferences
    if prefs is not None:
        bonus = 0.0
        # Cuisine match bonus
        if prefs.cuisine:
            cuisines_val = row.get(CleanColumns.CUISINES, "")
            cuisines_list_val = row.get(CleanColumns.CUISINES_LIST, [])
            c_low = prefs.cuisine.lower().strip()
            if hasattr(cuisines_list_val, "__iter__") and not isinstance(cuisines_list_val, str):
                if any(str(c).lower().strip() == c_low for c in cuisines_list_val):
                    bonus += 3.0
            elif c_low in str(cuisines_val).lower():
                bonus += 3.0

        # Additional preference tags bonus
        if prefs.preferences:
            combined_text = (
                f"{row.get(CleanColumns.RESTAURANT_TYPE, '')} "
                f"{row.get(CleanColumns.CUISINES, '')} "
                f"{row.get('dish_liked', '')}"
            ).lower()

            for p in prefs.preferences:
                p_clean = p.lower().strip()
                if p_clean in combined_text:
                    bonus += 1.5
                elif p_clean == "vegetarian" and "veg" in combined_text:
                    bonus += 1.5
                elif p_clean == "buffet" and "buffet" in combined_text:
                    bonus += 1.5
                elif p_clean in ("fine dining", "luxury") and "fine dining" in combined_text:
                    bonus += 2.0
                elif p_clean == "quick service" and "quick" in combined_text:
                    bonus += 1.5

        if bool(row.get(CleanColumns.ONLINE_ORDER, False)):
            bonus += 0.5
        if bool(row.get(CleanColumns.BOOK_TABLE, False)):
            bonus += 0.8

        score += bonus

    return round(score, 4)


def get_score_breakdown(row: pd.Series) -> dict:
    """
    Return a detailed breakdown of the ranking score for debugging.

    Args:
        row: A single row from the ranked DataFrame.

    Returns:
        Dictionary with score components.
    """
    rating = float(row.get(CleanColumns.RATING, 0) or 0)
    votes = int(row.get(CleanColumns.VOTES, 0) or 0)

    return {
        "rating": rating,
        "votes": votes,
        "popularity_factor": round(math.log2(votes + 2), 4),
        "total_score": round(rating * math.log2(votes + 2), 4),
    }
