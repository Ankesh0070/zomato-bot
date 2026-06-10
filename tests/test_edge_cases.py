import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from src.data.schema import CleanColumns
from src.input.models import UserPreferences
from src.recommendation.engine import get_recommendations
from src.recommendation import llm_client
from src.recommendation.llm_client import BaseLLMClient


class BadClient(BaseLLMClient):
    def __init__(self):
        super().__init__(model="bad-model", temperature=0.0, max_tokens=1000)

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        # Return invalid (non-JSON) response to force parse failure
        return "This is not JSON and will cause the parser to fail"

    def is_available(self) -> bool:
        return True


def test_no_results_returns_graceful_message():
    empty_df = pd.DataFrame(columns=[
        CleanColumns.RESTAURANT_NAME,
        CleanColumns.LOCATION,
        CleanColumns.RATING,
        CleanColumns.COST_FOR_TWO,
    ])

    prefs = UserPreferences(location="Nowhere")
    result = get_recommendations(empty_df, prefs, use_cache=False)

    assert result["candidates_found"] == 0
    assert result["model_used"] == "none"
    assert "No restaurants found" in result["result"].summary


def test_llm_parse_failure_triggers_rule_based(monkeypatch):
    # Create a small candidates dataframe
    df = pd.DataFrame([
        {
            CleanColumns.RESTAURANT_NAME: "RB1",
            CleanColumns.LOCATION: "TestCity",
            CleanColumns.RATING: 4.0,
            CleanColumns.COST_FOR_TWO: 600,
            CleanColumns.CUISINES: "Indian",
            CleanColumns.CUISINES_LIST: ["Indian"],
            CleanColumns.BUDGET_TIER: "medium",
        }
    ])

    prefs = UserPreferences(location="NonExistentCity", budget=1000, party_size=2)

    # Patch LLM to return invalid content
    import src.recommendation.engine as engine_module
    monkeypatch.setattr(engine_module, "create_llm_client", lambda: BadClient())

    result = get_recommendations(df, prefs, use_cache=False)

    assert result["model_used"] == "rule-based"
    assert result["result"].recommendations
