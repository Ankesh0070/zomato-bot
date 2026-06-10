"""
Response parser module.

Parses and validates the raw LLM response, extracting structured
recommendation data from potentially messy JSON output.

Handles edge cases:
- JSON wrapped in markdown code fences
- Extra text before/after JSON
- Wrong key names (adaptive mapping)
- Missing or extra fields
- Hallucinated restaurants
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.data.schema import CleanColumns

logger = logging.getLogger("zomato_recommender.recommendation.parser")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    """A single restaurant recommendation from the LLM."""

    rank: int
    restaurant_name: str
    cuisine: str
    rating: float
    estimated_cost: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "restaurant_name": self.restaurant_name,
            "cuisine": self.cuisine,
            "rating": self.rating,
            "estimated_cost": self.estimated_cost,
            "explanation": self.explanation,
        }


@dataclass
class RecommendationResult:
    """Complete recommendation response from the LLM."""

    recommendations: list[Recommendation] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ParseError(Exception):
    """Raised when the LLM response cannot be parsed."""
    pass


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------
def parse_response(
    raw_response: str,
    candidates_df: Optional[pd.DataFrame] = None,
    max_recommendations: int = 5,
) -> RecommendationResult:
    """
    Parse and validate an LLM response into a RecommendationResult.

    Steps:
        1. Extract JSON from raw response
        2. Validate and adapt the schema
        3. Build Recommendation objects
        4. Cross-reference against candidate data (hallucination check)
        5. Truncate to max_recommendations

    Args:
        raw_response: Raw text response from the LLM.
        candidates_df: DataFrame of candidates sent to the LLM (for validation).
        max_recommendations: Maximum recommendations to return.

    Returns:
        Validated RecommendationResult.

    Raises:
        ParseError: If the response cannot be parsed at all.
    """
    # Step 1: Extract JSON
    data = _extract_json(raw_response)

    # Step 2: Find recommendations array
    recommendations_raw = _find_recommendations_array(data)

    if not recommendations_raw:
        raise ParseError(
            "No recommendations found in LLM response. "
            f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
        )

    # Step 3: Build Recommendation objects
    recommendations = []
    for idx, item in enumerate(recommendations_raw[:max_recommendations], start=1):
        try:
            rec = _parse_single_recommendation(item, idx)
            recommendations.append(rec)
        except Exception as e:
            logger.warning(f"Skipping malformed recommendation #{idx}: {e}")

    # Step 4: Hallucination check
    if candidates_df is not None and not candidates_df.empty:
        recommendations = _check_hallucinations(recommendations, candidates_df)

    # Step 5: Re-rank after any filtering
    for i, rec in enumerate(recommendations, start=1):
        rec.rank = i

    # Extract summary
    summary = _extract_summary(data)

    result = RecommendationResult(
        recommendations=recommendations,
        summary=summary,
        raw_response=raw_response,
    )

    logger.info(f"Parsed {len(recommendations)} valid recommendations.")
    return result


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> dict:
    """
    Extract JSON from potentially messy LLM output.

    Tries multiple strategies in order:
        1. Direct JSON parse
        2. Extract from markdown code fences
        3. Find first { to last }
        4. Raise ParseError
    """
    raw = raw.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try to find a JSON array [ ... ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(raw[start : end + 1])
            return {"recommendations": arr}
        except json.JSONDecodeError:
            pass

    raise ParseError(
        f"Could not extract valid JSON from LLM response. "
        f"Response preview: {raw[:200]}..."
    )


# ---------------------------------------------------------------------------
# Schema Adaptation
# ---------------------------------------------------------------------------
def _find_recommendations_array(data: dict) -> list[dict]:
    """
    Find the recommendations array, handling alternative key names.

    Checks for: recommendations, results, restaurants, suggestions, data
    """
    if not isinstance(data, dict):
        if isinstance(data, list):
            return data
        return []

    # Try common key names
    candidate_keys = [
        "recommendations",
        "results",
        "restaurants",
        "suggestions",
        "data",
        "top_restaurants",
        "picks",
    ]

    for key in candidate_keys:
        if key in data and isinstance(data[key], list):
            return data[key]

    # Try to find any list value in the dict
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:
            # Check if items look like recommendations
            if isinstance(value[0], dict):
                logger.info(f"Found recommendations under non-standard key '{key}'")
                return value

    return []


def _extract_summary(data: dict) -> str:
    """Extract the summary string from the response."""
    if not isinstance(data, dict):
        return ""

    summary_keys = ["summary", "overall_summary", "conclusion", "overview"]
    for key in summary_keys:
        if key in data and isinstance(data[key], str):
            return data[key]

    return ""


# ---------------------------------------------------------------------------
# Single Recommendation Parser
# ---------------------------------------------------------------------------
def _parse_single_recommendation(item: dict, default_rank: int) -> Recommendation:
    """Parse a single recommendation dict into a Recommendation object."""
    if not isinstance(item, dict):
        raise ValueError(f"Expected dict, got {type(item)}")

    # Flexible field extraction
    name = _get_field(item, ["restaurant_name", "name", "restaurant", "title"], "Unknown")
    cuisine = _get_field(item, ["cuisine", "cuisines", "cuisine_type"], "N/A")
    rating = _get_numeric_field(item, ["rating", "rate", "score", "stars"], 0.0)
    cost = _get_field(
        item,
        ["estimated_cost", "cost", "price", "cost_for_two", "approx_cost"],
        "Price N/A",
    )
    explanation = _get_field(
        item,
        ["explanation", "reason", "why", "description", "rationale", "note"],
        "No explanation provided.",
    )
    rank = _get_numeric_field(item, ["rank", "position", "#"], default_rank)

    return Recommendation(
        rank=int(rank),
        restaurant_name=str(name),
        cuisine=str(cuisine),
        rating=float(rating),
        estimated_cost=str(cost),
        explanation=str(explanation),
    )


def _get_field(data: dict, keys: list[str], default: str) -> str:
    """Try multiple keys to extract a string value."""
    for key in keys:
        if key in data and data[key] is not None:
            return str(data[key])
    return default


def _get_numeric_field(data: dict, keys: list[str], default: float) -> float:
    """Try multiple keys to extract a numeric value."""
    for key in keys:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (ValueError, TypeError):
                continue
    return default


# ---------------------------------------------------------------------------
# Hallucination Check
# ---------------------------------------------------------------------------
def _check_hallucinations(
    recommendations: list[Recommendation],
    candidates_df: pd.DataFrame,
) -> list[Recommendation]:
    """
    Cross-reference recommendations against the candidate data.

    Removes recommendations that reference restaurants not in the
    candidate shortlist (i.e., hallucinated by the LLM).
    """
    if CleanColumns.RESTAURANT_NAME not in candidates_df.columns:
        return recommendations

    candidate_names = set(
        candidates_df[CleanColumns.RESTAURANT_NAME].str.lower().str.strip().tolist()
    )

    valid = []
    for rec in recommendations:
        rec_name_lower = rec.restaurant_name.lower().strip()

        if rec_name_lower in candidate_names:
            valid.append(rec)
        else:
            # Fuzzy check — the LLM might slightly modify the name
            best_match = _find_closest_name(rec_name_lower, candidate_names)
            if best_match:
                logger.info(
                    f"Corrected hallucinated name: '{rec.restaurant_name}' -> '{best_match}'"
                )
                rec.restaurant_name = best_match.title()
                valid.append(rec)
            else:
                logger.warning(
                    f"Removing hallucinated restaurant: '{rec.restaurant_name}' "
                    f"(not found in candidates)"
                )

    return valid


def _find_closest_name(
    name: str, candidate_names: set[str], threshold: float = 0.8
) -> Optional[str]:
    """Find the closest matching candidate name using simple ratio."""
    from fuzzywuzzy import fuzz

    best_match = None
    best_score = 0

    for candidate in candidate_names:
        score = fuzz.ratio(name, candidate) / 100.0
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate

    return best_match
