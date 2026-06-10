"""
Prompt builder module.

Constructs the multi-part LLM prompt from user preferences
and the filtered/ranked restaurant shortlist.

Prompt structure:
    Part 1 — System prompt (role, format, criteria)
    Part 2 — User context (preferences)
    Part 3 — Restaurant data (candidate list)
    Part 4 — Instructions (output format, constraints)
"""

import logging

from src.config import AppConfig

logger = logging.getLogger("zomato_recommender.recommendation.prompt_builder")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert restaurant recommendation advisor. Your role is to help users find the perfect dining experience based on their preferences.

Guidelines:
- Be professional, factual, and helpful.
- Base your recommendations ONLY on the restaurant data provided below. Do NOT invent or hallucinate restaurants.
- Rank restaurants considering how well they match the user's stated preferences.
- Provide a clear, concise explanation for each recommendation (2-3 sentences).
- Consider factors like cuisine match, rating, cost alignment with budget, and any additional preferences.
- If a restaurant does not perfectly match all criteria, acknowledge it and explain why it is still a good option.

Output Format:
You MUST respond with valid JSON only. No markdown, no code fences, no extra text.
Use exactly this schema:
{
  "recommendations": [
    {
      "rank": 1,
      "restaurant_name": "Exact name from the data",
      "cuisine": "Cuisine type(s)",
      "rating": 4.5,
      "estimated_cost": "Rs.XXX for two",
      "explanation": "2-3 sentence explanation of why this is recommended."
    }
  ],
  "summary": "A 2-3 sentence overall summary of the recommendations."
}"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------
def build_prompt(
    user_context: str,
    restaurant_data: str,
    max_recommendations: int | None = None,
) -> dict[str, str]:
    """
    Build the complete multi-part prompt for the LLM.

    Args:
        user_context: Formatted user preferences (from formatter).
        restaurant_data: Formatted restaurant list (from formatter).
        max_recommendations: Number of recommendations to request.

    Returns:
        Dictionary with 'system_prompt' and 'user_prompt' keys.
    """
    if max_recommendations is None:
        max_recommendations = AppConfig.MAX_RECOMMENDATIONS

    user_prompt = _build_user_prompt(
        user_context=user_context,
        restaurant_data=restaurant_data,
        max_recommendations=max_recommendations,
    )

    prompt = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }

    # Log token estimate
    total_chars = len(prompt["system_prompt"]) + len(prompt["user_prompt"])
    approx_tokens = total_chars // 4
    logger.info(f"Built prompt: ~{approx_tokens} tokens (system + user)")

    if approx_tokens > 3500:
        logger.warning(
            f"Prompt may be too long (~{approx_tokens} tokens). "
            f"Consider reducing candidate count."
        )

    return prompt


def _build_user_prompt(
    user_context: str,
    restaurant_data: str,
    max_recommendations: int,
) -> str:
    """Assemble the user-facing portion of the prompt."""

    instructions = (
        f"\nInstructions:\n"
        f"- Select the top {max_recommendations} restaurants from the data above.\n"
        f"- Rank them from best match (#1) to good match (#{max_recommendations}).\n"
        f"- For each restaurant, provide: rank, restaurant_name (exact match from data), "
        f"cuisine, rating, estimated_cost, and a 2-3 sentence explanation.\n"
        f"- Also provide a 2-3 sentence 'summary' of your overall recommendations.\n"
        f"- Respond with valid JSON only. No markdown, no code fences.\n"
        f"- Use ONLY restaurants from the provided data. Do NOT invent restaurants."
    )

    return f"{user_context}\n\n{restaurant_data}\n{instructions}"


def build_retry_prompt(
    original_prompt: dict[str, str],
    error_message: str,
) -> dict[str, str]:
    """
    Build a follow-up prompt for when the LLM returns invalid output.

    Appends stricter instructions to the original prompt.

    Args:
        original_prompt: The original prompt dict.
        error_message: Description of what went wrong.

    Returns:
        Updated prompt dict with stricter instructions.
    """
    retry_suffix = (
        f"\n\nIMPORTANT: Your previous response was invalid. "
        f"Error: {error_message}\n"
        f"Please respond with ONLY valid JSON. No explanations, no markdown code fences, "
        f"no extra text before or after the JSON. "
        f"Start your response with {{ and end with }}."
    )

    return {
        "system_prompt": original_prompt["system_prompt"],
        "user_prompt": original_prompt["user_prompt"] + retry_suffix,
    }


def estimate_prompt_tokens(prompt: dict[str, str]) -> int:
    """Estimate the total token count of a prompt."""
    total = sum(len(v) for v in prompt.values())
    return total // 4
