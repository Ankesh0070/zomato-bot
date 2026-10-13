"""
LLM context formatter module.

Converts the ranked restaurant shortlist and user preferences into
structured text suitable for embedding in an LLM prompt.
"""

import logging

import pandas as pd

from src.data.schema import CleanColumns
from src.input.models import UserPreferences

logger = logging.getLogger("zomato_recommender.integration.formatter")

# Approximate token budget for restaurant data in prompt
MAX_RESTAURANTS_IN_PROMPT = 20
APPROX_TOKENS_PER_RESTAURANT = 60  # ~60 tokens per formatted entry


def format_for_llm(
    df: pd.DataFrame,
    prefs: UserPreferences,
    max_restaurants: int = MAX_RESTAURANTS_IN_PROMPT,
) -> dict[str, int | str]:
    """
    Format the ranked restaurant shortlist and user preferences
    into structured text blocks for the LLM prompt.

    Args:
        df: Ranked DataFrame of restaurant candidates.
        prefs: User preferences.
        max_restaurants: Maximum number of restaurants to include.

    Returns:
        Dictionary with keys:
            - "user_context": Formatted user preferences text.
            - "restaurant_data": Formatted restaurant list text.
            - "candidate_count": Number of candidates included.
    """
    # Limit the number of restaurants
    df_limited = df.head(max_restaurants)

    user_context = _format_user_preferences(prefs)
    restaurant_data = _format_restaurant_list(df_limited)

    result = {
        "user_context": user_context,
        "restaurant_data": restaurant_data,
        "candidate_count": len(df_limited),
    }

    # Estimate token count
    total_text = user_context + restaurant_data
    approx_tokens = len(total_text) // 4
    logger.info(
        f"Formatted {len(df_limited)} restaurants for LLM "
        f"(~{approx_tokens} tokens)"
    )

    return result


def _format_user_preferences(prefs: UserPreferences) -> str:
    """Format user preferences into natural language context."""
    budget_desc = f"₹{prefs.budget} total for {prefs.party_size} {'person' if prefs.party_size == 1 else 'people'}" if prefs.budget else "Flexible (no specific limit)"
    lines = [
        "User Preferences:",
        f"- Location: {prefs.location}",
        f"- Party Size: {prefs.party_size} {'person' if prefs.party_size == 1 else 'people'}",
        f"- Budget: {budget_desc}",
    ]

    if prefs.cuisine:
        lines.append(f"- Preferred Cuisine: {prefs.cuisine}")
    else:
        lines.append("- Preferred Cuisine: No specific preference")

    lines.append(f"- Minimum Rating: {prefs.min_rating}/5")

    if prefs.preferences:
        lines.append(f"- Additional Preferences: {', '.join(prefs.preferences)}")

    return "\n".join(lines)


def _format_restaurant_list(df: pd.DataFrame) -> str:
    """Format the restaurant shortlist into a numbered list."""
    if df.empty:
        return "No matching restaurants found."

    lines = [f"\nRestaurant Data ({len(df)} candidates):"]

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        entry = _format_single_restaurant(row, idx)
        lines.append(entry)

    return "\n".join(lines)


def _format_single_restaurant(row: pd.Series, index: int) -> str:
    """Format a single restaurant row into a compact text line."""
    name = row.get(CleanColumns.RESTAURANT_NAME, "Unknown")
    cuisine = row.get(CleanColumns.CUISINES, "N/A")
    rating = row.get(CleanColumns.RATING, "N/A")
    cost = row.get(CleanColumns.COST_FOR_TWO, "N/A")
    votes = row.get(CleanColumns.VOTES, 0)
    online_order = row.get(CleanColumns.ONLINE_ORDER, False)
    book_table = row.get(CleanColumns.BOOK_TABLE, False)
    rest_type = row.get(CleanColumns.RESTAURANT_TYPE, "")

    # Format rating
    if pd.notna(rating):
        rating_str = f"{float(rating):.1f}/5"
    else:
        rating_str = "N/A"

    # Format cost
    if pd.notna(cost):
        cost_str = f"Rs.{int(cost)} for two"
    else:
        cost_str = "Price N/A"

    # Format votes
    if pd.notna(votes):
        votes_str = f"{int(votes):,}"
    else:
        votes_str = "0"

    # Build the line
    parts = [
        f"{index}. Name: {name}",
        f"Cuisine: {cuisine}",
        f"Rating: {rating_str}",
        f"Cost: {cost_str}",
        f"Votes: {votes_str}",
    ]

    # Optional fields
    if online_order:
        parts.append("Online Order: Yes")
    if book_table:
        parts.append("Book Table: Yes")
    if rest_type:
        parts.append(f"Type: {rest_type}")

    return " | ".join(parts)


def estimate_token_count(formatted: dict[str, int | str]) -> int:
    """
    Estimate the token count of the formatted content.

    Uses a rough approximation of ~4 characters per token.

    Args:
        formatted: Output from format_for_llm().

    Returns:
        Estimated token count.
    """
    total_chars = sum(len(v) for v in formatted.values() if isinstance(v, str))
    return total_chars // 4
