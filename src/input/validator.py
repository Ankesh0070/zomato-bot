"""
Input validation and normalization module.

Validates and sanitizes user input to ensure it conforms to expected
formats and is safe to use in the filtering and LLM pipelines.
"""

import logging
import re
from typing import Optional

from fuzzywuzzy import process as fuzz_process

from src.input.models import UserPreferences, VALID_BUDGETS

logger = logging.getLogger("zomato_recommender.input.validator")

# Maximum allowed input lengths
MAX_TEXT_LENGTH = 100
MAX_PREFERENCES_COUNT = 5

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"act\s+as",
    r"forget\s+(everything|all)",
    r"override\s+(all|system)",
    r"new\s+instructions",
]

# Budget aliases for fuzzy matching
BUDGET_ALIASES = {
    "cheap": "low",
    "affordable": "low",
    "inexpensive": "low",
    "budget": "low",
    "moderate": "medium",
    "mid": "medium",
    "mid-range": "medium",
    "average": "medium",
    "expensive": "high",
    "premium": "high",
    "luxury": "high",
    "upscale": "high",
    "fine": "high",
}


def validate_and_normalize(
    location: str,
    budget: int | None = None,
    party_size: int = 2,
    cuisine: Optional[str] = None,
    min_rating: float = 3.5,
    preferences: Optional[list[str]] = None,
    known_locations: Optional[list[str]] = None,
    known_cuisines: Optional[list[str]] = None,
) -> tuple[UserPreferences, list[str]]:
    """
    Validate and normalize user input into a UserPreferences object.

    Args:
        location: Raw location input from user.
        budget: Raw numeric budget from user.
        party_size: Number of people in the party.
        cuisine: Raw cuisine input from user (optional).
        min_rating: Raw minimum rating input from user.
        preferences: Raw preference tags from user.
        known_locations: List of valid locations from the dataset.
        known_cuisines: List of valid cuisines from the dataset.

    Returns:
        Tuple of (validated UserPreferences, list of warning messages).
        Warnings are non-fatal issues like auto-corrections.

    Raises:
        ValueError: If a critical input is missing or invalid.
    """
    warnings: list[str] = []

    # --- Location ---
    location, loc_warnings = _validate_location(location, known_locations)
    warnings.extend(loc_warnings)

    # --- Budget ---
    budget, budget_warnings = _validate_budget(budget)
    warnings.extend(budget_warnings)
    
    # --- Party Size ---
    party_size, party_warnings = _validate_party_size(party_size)
    warnings.extend(party_warnings)

    # --- Cuisine ---
    cuisine, cuisine_warnings = _validate_cuisine(cuisine, known_cuisines)
    warnings.extend(cuisine_warnings)

    # --- Rating ---
    min_rating, rating_warnings = _validate_rating(min_rating)
    warnings.extend(rating_warnings)

    # --- Preferences ---
    if preferences is None:
        preferences = []
    preferences = _validate_preferences(preferences)

    prefs = UserPreferences(
        location=location,
        budget=budget,
        party_size=party_size,
        cuisine=cuisine,
        min_rating=min_rating,
        preferences=preferences,
    )

    logger.info(f"Validated preferences: {prefs}")
    return prefs, warnings


# ---------------------------------------------------------------------------
# Location validation
# ---------------------------------------------------------------------------
def _validate_location(
    location: str,
    known_locations: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    """Validate and normalize location input."""
    warnings: list[str] = []

    # Sanitize
    location = _sanitize_text(location)

    if not location:
        raise ValueError("Location is required. Please specify a city or area.")

    # Truncate if too long
    if len(location) > MAX_TEXT_LENGTH:
        location = location[:MAX_TEXT_LENGTH]
        warnings.append(f"Location truncated to {MAX_TEXT_LENGTH} characters.")

    # Title case
    location = location.strip().title()

    # Fuzzy match against known locations
    if known_locations and location not in known_locations:
        match, score = _fuzzy_match(location, known_locations)
        if match and score >= 80:
            if score < 100:
                warnings.append(f"Showing results for '{match}' (did you mean?)")
            location = match
        else:
            # Suggest closest matches
            suggestions = _get_top_suggestions(location, known_locations, n=3)
            if suggestions:
                suggestion_str = ", ".join(suggestions)
                warnings.append(
                    f"'{location}' not found. Did you mean: {suggestion_str}?"
                )
            else:
                warnings.append(
                    f"'{location}' not found in our dataset. "
                    f"Results may be limited."
                )

    return location, warnings


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------
def _validate_budget(budget) -> tuple[int | None, list[str]]:
    """Validate and normalize numeric budget input."""
    warnings: list[str] = []

    if budget is None or str(budget).strip() == "":
        return None, warnings

    try:
        b = int(budget)
        if b < 0:
            warnings.append("Budget cannot be negative. Ignoring budget constraint.")
            return None, warnings
        return b, warnings
    except (ValueError, TypeError):
        warnings.append(f"Invalid budget '{budget}'. Ignoring budget constraint.")
        return None, warnings

def _validate_party_size(party_size) -> tuple[int, list[str]]:
    """Validate party size input."""
    warnings: list[str] = []
    
    if party_size is None or str(party_size).strip() == "":
        return 2, warnings
        
    try:
        p = int(party_size)
        if p < 1:
            warnings.append("Party size must be at least 1. Using default 2.")
            return 2, warnings
        return p, warnings
    except (ValueError, TypeError):
        warnings.append(f"Invalid party size '{party_size}'. Using default 2.")
        return 2, warnings


# ---------------------------------------------------------------------------
# Cuisine validation
# ---------------------------------------------------------------------------
def _validate_cuisine(
    cuisine: Optional[str],
    known_cuisines: Optional[list[str]] = None,
) -> tuple[Optional[str], list[str]]:
    """Validate and normalize cuisine input."""
    warnings: list[str] = []

    if cuisine is None or str(cuisine).strip() == "":
        return None, warnings

    cuisine = _sanitize_text(cuisine).strip().title()

    if len(cuisine) > MAX_TEXT_LENGTH:
        cuisine = cuisine[:MAX_TEXT_LENGTH]

    # Fuzzy match against known cuisines
    if known_cuisines and cuisine not in known_cuisines:
        match, score = _fuzzy_match(cuisine, known_cuisines)
        if match and score >= 80:
            if score < 100:
                warnings.append(f"Cuisine matched to '{match}'.")
            cuisine = match
        else:
            suggestions = _get_top_suggestions(cuisine, known_cuisines, n=3)
            if suggestions:
                suggestion_str = ", ".join(suggestions)
                warnings.append(
                    f"'{cuisine}' not available. Similar: {suggestion_str}"
                )

    return cuisine, warnings


# ---------------------------------------------------------------------------
# Rating validation
# ---------------------------------------------------------------------------
def _validate_rating(min_rating) -> tuple[float, list[str]]:
    """Validate and clamp rating to 0.0–5.0 range."""
    warnings: list[str] = []

    try:
        min_rating = float(min_rating)
    except (ValueError, TypeError):
        warnings.append(
            f"Invalid rating '{min_rating}'. Using default 3.5. "
            f"Rating must be a number between 0 and 5."
        )
        return 3.5, warnings

    original = min_rating

    # Clamp to valid range
    min_rating = max(0.0, min(5.0, min_rating))

    if min_rating != original:
        warnings.append(
            f"Rating clamped from {original} to {min_rating} "
            f"(valid range: 0.0–5.0)."
        )

    # Round to 1 decimal
    min_rating = round(min_rating, 1)

    return min_rating, warnings


# ---------------------------------------------------------------------------
# Preferences validation
# ---------------------------------------------------------------------------
def _validate_preferences(preferences: list[str]) -> list[str]:
    """Sanitize and limit preference tags."""
    cleaned = []
    for pref in preferences:
        pref = _sanitize_text(str(pref)).strip().lower()
        if pref and len(pref) <= MAX_TEXT_LENGTH:
            cleaned.append(pref)

    # Limit count
    return cleaned[:MAX_PREFERENCES_COUNT]


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------
def _sanitize_text(text: str) -> str:
    """
    Sanitize text input: remove injection patterns and dangerous characters.

    Strips HTML tags, known prompt injection patterns, and limits to
    safe alphanumeric + common punctuation characters.
    """
    if not text:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove prompt injection patterns
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Keep only safe characters: letters, digits, spaces, hyphens, commas, periods
    text = re.sub(r"[^\w\s\-,.'()]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: int = 60,
) -> tuple[Optional[str], int]:
    """
    Find the best fuzzy match for a query in a list of candidates.

    Returns:
        Tuple of (best_match, score). Returns (None, 0) if no match above threshold.
    """
    if not candidates:
        return None, 0

    result = fuzz_process.extractOne(query, candidates)
    if result and result[1] >= threshold:
        return result[0], result[1]
    return None, 0


def _get_top_suggestions(
    query: str,
    candidates: list[str],
    n: int = 3,
) -> list[str]:
    """Return top-N fuzzy match suggestions for a query."""
    if not candidates:
        return []

    results = fuzz_process.extract(query, candidates, limit=n)
    return [r[0] for r in results if r[1] >= 50]
