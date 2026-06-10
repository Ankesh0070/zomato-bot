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

logger = logging.getLogger("zomato_recommender.integration.ranker")


def rank_and_select(
    df: pd.DataFrame,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Rank restaurants by composite score and return the top-K.

    The composite score balances:
        - Rating (primary signal)
        - Popularity (log of votes — diminishing returns)

    Score formula:
        score = rating * log2(votes + 2)

    Using log2(votes + 2) ensures:
        - Restaurants with 0 votes get a non-zero multiplier (~1.0)
        - High vote counts have diminishing influence
        - Rating remains the dominant factor

    Args:
        df: Filtered restaurant DataFrame.
        top_k: Number of top candidates to return. Defaults to AppConfig.TOP_K_CANDIDATES.

    Returns:
        Top-K ranked DataFrame sorted by score (descending).
    """
    if top_k is None:
        top_k = AppConfig.TOP_K_CANDIDATES

    if df.empty:
        logger.warning("Empty DataFrame passed to ranker.")
        return df

    df = df.copy()

    # Calculate composite score
    df["_score"] = df.apply(_compute_score, axis=1)

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


def _compute_score(row: pd.Series) -> float:
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
