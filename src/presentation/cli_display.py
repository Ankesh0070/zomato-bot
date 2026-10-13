import sys

# Ensure UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.recommendation.parser import RecommendationResult


def render_header():
    """Render the CLI header."""
    print("=" * 60)
    print("🍽️  Zomato AI Recommender (CLI)")
    print("=" * 60)

def render_summary(result: RecommendationResult, metadata: dict):
    """Render the AI summary and metadata."""
    print(f"\n🤖 AI Summary: {result.summary}")
    print(f"Model: {result.model_used or 'unknown'} | Candidates: {metadata.get('candidates_found', 0)} | Time: {metadata.get('response_time_ms', 0)}ms\n")

def render_recommendations(result: RecommendationResult):
    """Render all recommendation cards."""
    for rec in result.recommendations:
        _render_single_card(rec)

def _render_single_card(rec):
    """Render a single recommendation card."""
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rec.rank, f"#{rec.rank}")
    print("-" * 60)
    print(f"{rank_emoji} {rec.restaurant_name}")
    print(f"🍕 Cuisine: {rec.cuisine} | ⭐ {rec.rating:.1f}/5 | 💰 {rec.estimated_cost}")
    print(f"💡 {rec.explanation}")

def render_error(error_message: str):
    """Render an error state."""
    print(f"\n❌ Error: {error_message}")
    print("Try again in a moment. If the issue persists, check your API key in `.env`.")

def render_no_results(location: str):
    """Render the no-results state."""
    print(f"\n😕 No restaurants found in {location}.")
    print("Try relaxing your filters — broaden the cuisine, lower the rating, or change the budget.")
