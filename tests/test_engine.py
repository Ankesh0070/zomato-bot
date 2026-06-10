import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from src.data.schema import CleanColumns
from src.input.models import UserPreferences
from src.recommendation.engine import get_recommendations, clear_cache
from src.recommendation import llm_client
from src.recommendation.llm_client import BaseLLMClient


class DummyClient(BaseLLMClient):
    def __init__(self):
        super().__init__(model="dummy-model", temperature=0.0, max_tokens=1000)

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        # Return a valid JSON response the parser can consume
        return json.dumps({
            "recommendations": [
                {
                    "rank": 1,
                    "restaurant_name": "R1",
                    "cuisine": "Indian",
                    "rating": 4.5,
                    "estimated_cost": "Rs.500 for two",
                    "explanation": "Great choice!",
                }
            ],
            "summary": "Top pick from test dataset.",
        })

    def is_available(self) -> bool:
        return True


def test_get_recommendations_integration(monkeypatch):
    # Small test dataset
    df = pd.DataFrame([
        {
            CleanColumns.RESTAURANT_NAME: "R1",
            CleanColumns.LOCATION: "TestCity",
            CleanColumns.RATING: 4.5,
            CleanColumns.COST_FOR_TWO: 500,
            CleanColumns.CUISINES: "Indian",
            CleanColumns.CUISINES_LIST: ["Indian"],
            CleanColumns.BUDGET_TIER: "medium",
        }
    ])

    prefs = UserPreferences(location="TestCity", budget=500, party_size=2, cuisine="Indian", min_rating=3.0)

    # Patch LLM client factory to return our dummy client
    import src.recommendation.engine as engine_module
    monkeypatch.setattr(engine_module, "create_llm_client", lambda: DummyClient())

    clear_cache()

    result = get_recommendations(df, prefs, use_cache=True)

    assert "result" in result
    assert result["result"].recommendations
    assert result["cached"] is False
    assert result["model_used"] is not None

    # Second call should hit cache
    result2 = get_recommendations(df, prefs, use_cache=True)
    assert result2["cached"] is True
