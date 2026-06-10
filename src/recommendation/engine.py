"""
Recommendation engine — orchestrates the full recommendation pipeline.

Coordinates: prompt building -> LLM call -> response parsing.
Handles retries, fallback providers, and result caching.
"""

import hashlib
import json
import logging
import time
from typing import Optional

import pandas as pd

from src.config import AppConfig
from src.input.models import UserPreferences
from src.integration.filter import apply_filters
from src.integration.formatter import format_for_llm
from src.integration.ranker import rank_and_select
from src.recommendation.llm_client import (
    BaseLLMClient,
    LLMError,
    LLMAuthError,
    call_with_retry,
    create_llm_client,
)
from src.recommendation.parser import (
    ParseError,
    RecommendationResult,
    parse_response,
)
from src.recommendation.prompt_builder import (
    build_prompt,
    build_retry_prompt,
)

logger = logging.getLogger("zomato_recommender.recommendation.engine")


# ---------------------------------------------------------------------------
# Response Cache
# ---------------------------------------------------------------------------
_response_cache: dict[str, tuple[RecommendationResult, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_key(prefs: UserPreferences) -> str:
    """Generate a deterministic cache key from user preferences."""
    data = json.dumps(prefs.to_dict(), sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()


def _get_cached(prefs: UserPreferences) -> Optional[RecommendationResult]:
    """Retrieve a cached result if it exists and hasn't expired."""
    key = _cache_key(prefs)
    if key in _response_cache:
        result, timestamp = _response_cache[key]
        age = time.time() - timestamp
        if age < CACHE_TTL_SECONDS:
            logger.info(f"Cache hit (age: {age:.0f}s)")
            return result
        else:
            del _response_cache[key]
            logger.debug("Cache entry expired, removed.")
    return None


def _set_cache(prefs: UserPreferences, result: RecommendationResult) -> None:
    """Store a result in the cache."""
    key = _cache_key(prefs)
    _response_cache[key] = (result, time.time())
    logger.debug(f"Result cached (key: {key[:8]}...)")


def clear_cache() -> None:
    """Clear the entire response cache."""
    _response_cache.clear()
    logger.info("Response cache cleared.")


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
def get_recommendations(
    df: pd.DataFrame,
    prefs: UserPreferences,
    use_cache: bool = True,
) -> dict:
    """
    Run the full recommendation pipeline: filter -> rank -> prompt -> LLM -> parse.

    Args:
        df: Preprocessed restaurant DataFrame.
        prefs: Validated user preferences.
        use_cache: Whether to use response caching.

    Returns:
        Dictionary with:
            - 'result': RecommendationResult object
            - 'metadata': Filter/ranking metadata
            - 'model_used': Which LLM model was used
            - 'response_time_ms': Total time in milliseconds
            - 'candidates_found': Number of candidates after filtering
    """
    start_time = time.time()

    # Check cache
    if use_cache:
        cached = _get_cached(prefs)
        if cached:
            elapsed = (time.time() - start_time) * 1000
            return {
                "result": cached,
                "metadata": {"source": "cache"},
                "model_used": cached.model_used,
                "response_time_ms": round(elapsed),
                "candidates_found": 0,
                "cached": True,
            }

    # Step 1: Filter
    logger.info(f"Step 1/4: Filtering restaurants for {prefs.location}...")
    filtered, filter_metadata = apply_filters(df, prefs)
    logger.info(f"  -> {len(filtered)} candidates after filtering")

    if filtered.empty:
        logger.warning("No restaurants match the filters.")
        elapsed = (time.time() - start_time) * 1000
        return {
            "result": RecommendationResult(
                summary=f"No restaurants found matching your criteria in {prefs.location}. "
                        f"Try relaxing your filters."
            ),
            "metadata": filter_metadata,
            "model_used": "none",
            "response_time_ms": round(elapsed),
            "candidates_found": 0,
            "cached": False,
        }

    # Step 2: Rank
    logger.info("Step 2/4: Ranking candidates...")
    ranked = rank_and_select(filtered, top_k=AppConfig.TOP_K_CANDIDATES)
    logger.info(f"  -> Top {len(ranked)} candidates selected")

    # Step 3: Format + Prompt
    logger.info("Step 3/4: Building LLM prompt...")
    formatted = format_for_llm(ranked, prefs)
    prompt = build_prompt(
        user_context=formatted["user_context"],
        restaurant_data=formatted["restaurant_data"],
        max_recommendations=AppConfig.MAX_RECOMMENDATIONS,
    )

    # Step 4: LLM Call + Parse
    logger.info("Step 4/4: Calling LLM...")
    result, model_used = _call_llm_with_fallback(prompt, ranked)

    result.model_used = model_used

    # Cache the result
    if use_cache:
        _set_cache(prefs, result)

    elapsed = (time.time() - start_time) * 1000
    logger.info(
        f"Pipeline complete: {len(result.recommendations)} recommendations "
        f"in {elapsed:.0f}ms (model: {model_used})"
    )

    return {
        "result": result,
        "metadata": filter_metadata,
        "model_used": model_used,
        "response_time_ms": round(elapsed),
        "candidates_found": len(filtered),
        "cached": False,
    }


# ---------------------------------------------------------------------------
# LLM Call with Fallback + Parse
# ---------------------------------------------------------------------------
def _call_llm_with_fallback(
    prompt: dict[str, str],
    candidates_df: pd.DataFrame,
) -> tuple[RecommendationResult, str]:
    """
    Call the LLM with retry + provider fallback, then parse the response.

    Attempt sequence:
        1. Primary provider with retries
        2. If parse fails: re-prompt once with stricter instructions
        3. If primary provider fails entirely: try fallback provider
        4. If everything fails: return rule-based fallback

    Returns:
        Tuple of (RecommendationResult, model_name).
    """
    # Try Groq provider
    try:
        client = create_llm_client()
        result, model = _try_provider(client, prompt, candidates_df)
        if result:
            return result, model
    except LLMAuthError as e:
        logger.warning(f"Groq auth error: {e}")
    except LLMError as e:
        logger.warning(f"Groq provider failed: {e}")

    # Rule-based fallback
    logger.warning("LLM provider failed. Using rule-based fallback.")
    return _rule_based_fallback(candidates_df), "rule-based"


def _try_provider(
    client: BaseLLMClient,
    prompt: dict[str, str],
    candidates_df: pd.DataFrame,
) -> tuple[Optional[RecommendationResult], str]:
    """
    Try a single LLM provider: call -> parse, with one retry on parse failure.
    """
    model_name = f"{client.__class__.__name__}({client.model})"

    # First attempt
    try:
        raw_response = call_with_retry(
            client,
            user_prompt=prompt["user_prompt"],
            system_prompt=prompt["system_prompt"],
        )

        result = parse_response(raw_response, candidates_df)
        if result.recommendations:
            return result, model_name

    except ParseError as e:
        logger.warning(f"Parse error on first attempt: {e}")

        # Retry with stricter prompt
        try:
            retry_prompt = build_retry_prompt(prompt, str(e))
            raw_response = client.generate(
                user_prompt=retry_prompt["user_prompt"],
                system_prompt=retry_prompt["system_prompt"],
            )
            result = parse_response(raw_response, candidates_df)
            if result.recommendations:
                return result, model_name
        except (ParseError, LLMError) as e2:
            logger.warning(f"Retry also failed: {e2}")

    except LLMError:
        raise  # Let caller handle

    return None, model_name


# ---------------------------------------------------------------------------
# Rule-based Fallback
# ---------------------------------------------------------------------------
def _rule_based_fallback(candidates_df: pd.DataFrame) -> RecommendationResult:
    """
    Generate recommendations using simple rule-based ranking
    when all LLM providers are unavailable.

    Uses the already-ranked candidates and generates generic explanations.
    """
    from src.recommendation.parser import Recommendation

    recommendations = []
    max_recs = AppConfig.MAX_RECOMMENDATIONS

    for idx, (_, row) in enumerate(candidates_df.head(max_recs).iterrows(), start=1):
        name = row.get("restaurant_name", "Unknown")
        cuisine = row.get("cuisines", "N/A")
        rating = float(row.get("rating", 0) or 0)
        cost = row.get("cost_for_two", 0)
        cost_str = f"Rs.{int(cost)} for two" if pd.notna(cost) else "Price N/A"

        explanation = (
            f"Ranked #{idx} based on its {rating:.1f}/5 rating and popularity. "
            f"Serves {cuisine} cuisine."
        )

        recommendations.append(
            Recommendation(
                rank=idx,
                restaurant_name=str(name),
                cuisine=str(cuisine),
                rating=rating,
                estimated_cost=cost_str,
                explanation=explanation,
            )
        )

    return RecommendationResult(
        recommendations=recommendations,
        summary=(
            f"These {len(recommendations)} restaurants were selected based on "
            f"rating and popularity. (AI explanations unavailable — "
            f"using rule-based ranking.)"
        ),
    )
