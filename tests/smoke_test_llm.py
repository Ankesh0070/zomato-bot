"""Quick smoke test: verify Groq LLM call works end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.recommendation.llm_client import call_with_retry, create_llm_client

setup_logging()

print("=" * 60)
print("  Groq LLM Smoke Test")
print("=" * 60)

client = create_llm_client()
print(f"  Provider : {client}")
print(f"  Available: {client.is_available()}")

print("\n  Sending test prompt...")
response = call_with_retry(
    client,
    user_prompt='Recommend 1 Indian restaurant. Respond ONLY with JSON: {"recommendations": [{"rank": 1, "restaurant_name": "Example", "cuisine": "North Indian", "rating": 4.5, "estimated_cost": "Rs.500 for two", "explanation": "Great food."}], "summary": "Top pick."}',
    system_prompt="You are a restaurant advisor. Respond with valid JSON only.",
    max_retries=2,
)

print(f"\n  [OK] Response received ({len(response)} chars)")
print("\n  Response:")
print(f"  {response[:500]}")
print("\n" + "=" * 60)
print("  Groq LLM connection is WORKING!")
print("=" * 60)
