import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging, validate_config
from src.data.ingestion import download_dataset
from src.data.preprocessing import (
    get_unique_cuisines,
    get_unique_locations,
    preprocess,
)
from src.input.validator import validate_and_normalize
from src.presentation.cli_display import (
    render_error,
    render_header,
    render_no_results,
    render_recommendations,
    render_summary,
)
from src.recommendation.engine import get_recommendations


def main():
    parser = argparse.ArgumentParser(description="Zomato AI Recommender CLI")
    parser.add_argument("--location", type=str, required=True, help="Location to search (e.g. Delhi)")
    parser.add_argument("--budget", type=int, default=1000, help="Total budget in Rupees")
    parser.add_argument("--party_size", type=int, default=2, help="Number of people (optional)")
    parser.add_argument("--cuisine", type=str, default=None, help="Cuisine type")
    parser.add_argument("--min_rating", type=float, default=3.5, help="Minimum rating (0-5)")
    parser.add_argument("--preferences", type=str, nargs="*", default=[], help="Additional preferences")
    
    args = parser.parse_args()

    # Ensure UTF-8 output on Windows terminal
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    render_header()
    
    config_warnings = validate_config()
    if config_warnings:
        for w in config_warnings:
            print(f"⚠️ Warning: {w}")

    print("Loading dataset...")
    try:
        import pandas as pd
        clean_path = Path(__file__).resolve().parent.parent / "data" / "restaurants_clean.parquet"
        if clean_path.exists():
            df = pd.read_parquet(clean_path)
            if "cuisines_list" in df.columns:
                df["cuisines_list"] = df["cuisines_list"].apply(
                    lambda x: [str(c).strip() for c in x if c and str(c).strip() and str(c).lower() != "nan"]
                    if hasattr(x, "__iter__") and not isinstance(x, str)
                    else []
                )
        else:
            raw_df = download_dataset()
            df = preprocess(raw_df)
    except Exception as e:
        render_error(f"Failed to load dataset: {e}")
        return

    known_locations = get_unique_locations(df)
    known_cuisines = get_unique_cuisines(df)

    try:
        validated, validation_warnings = validate_and_normalize(
            location=args.location,
            budget=args.budget,
            party_size=args.party_size,
            cuisine=args.cuisine,
            min_rating=args.min_rating,
            preferences=args.preferences,
            known_locations=known_locations,
            known_cuisines=known_cuisines,
        )
    except ValueError as e:
        render_error(str(e))
        return

    for w in validation_warnings:
        print(f"⚠️ Warning: {w}")

    print("🤖 AI is analyzing restaurants...\n")
    try:
        response = get_recommendations(df, validated)
    except Exception as e:
        render_error(str(e))
        return

    result = response.get("result")

    if result is None or not result.recommendations:
        render_no_results(validated.location)
        return

    render_summary(result, response)
    render_recommendations(result)

if __name__ == "__main__":
    setup_logging()
    main()
