"""
# ruff: noqa: E402
Zomato AI Restaurant Recommender — Streamlit App

Main entry point. Orchestrates:
    1. Dataset loading (cached)
    2. Sidebar preference collection
    3. Full recommendation pipeline (filter -> rank -> LLM -> parse)
    4. Results display with premium UI
"""

import logging
import streamlit as st

# -- Page config must be the FIRST Streamlit command --
st.set_page_config(
    page_title="Zomato AI Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered restaurant recommendations using Groq (Llama 3.3) and Zomato data from Hugging Face.",
    },
)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging, validate_config  # noqa: E402
from src.data.ingestion import download_dataset  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.input.validator import validate_and_normalize  # noqa: E402
from src.recommendation.engine import get_recommendations  # noqa: E402
from src.presentation.styles import get_custom_css  # noqa: E402
from src.presentation.renderer import (  # noqa: E402
    render_header,
    render_sidebar,
    render_summary,
    render_recommendations,
    render_warnings,
    render_empty_state,
    render_no_results,
    render_error,
    render_footer,
)

# Setup logging
setup_logging()
logger = logging.getLogger("zomato_recommender.main")


# ===================================================================
# Data Loading (cached across reruns)
# ===================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess():
    """Download, cache, and preprocess the Zomato dataset."""
    logger.info("Loading dataset...")
    raw_df = download_dataset()
    logger.info(f"Raw dataset: {len(raw_df)} rows")

    clean_df = preprocess(raw_df)
    logger.info(f"Clean dataset: {len(clean_df)} rows")
    return clean_df


# ===================================================================
# Main App
# ===================================================================
def main():
    """Main application logic."""

    # Inject custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Header
    render_header()

    # Validate config
    config_warnings = validate_config()
    if config_warnings:
        render_warnings(config_warnings)

    # Load data
    with st.spinner("Loading restaurant data..."):
        try:
            df = load_and_preprocess()
        except Exception as e:
            render_error(f"Failed to load dataset: {e}")
            logger.exception("Dataset loading failed")
            return

    # Sidebar inputs
    inputs = render_sidebar(df)

    # Main content area
    if inputs["submitted"]:
        _handle_recommendation_request(df, inputs)
    elif "last_result" in st.session_state:
        # Show previous results on reruns
        _display_results(
            st.session_state["last_result"],
            st.session_state["last_metadata"],
        )
    else:
        render_empty_state()

    # Footer
    render_footer()


# ===================================================================
# Recommendation Flow
# ===================================================================
def _handle_recommendation_request(df, inputs: dict):
    """Handle the full recommendation flow when user clicks the button."""

    # Validate & normalize using the validator (with dataset context)
    from src.data.preprocessing import get_unique_locations, get_unique_cuisines
    known_locations = get_unique_locations(df)
    known_cuisines = get_unique_cuisines(df)

    try:
        validated, validation_warnings = validate_and_normalize(
            location=inputs["location"],
            budget=inputs["budget"],
            party_size=inputs["party_size"],
            cuisine=inputs["cuisine"],
            min_rating=inputs["min_rating"],
            preferences=inputs["preferences"],
            known_locations=known_locations,
            known_cuisines=known_cuisines,
        )
    except ValueError as e:
        render_error(str(e))
        return

    # Run the pipeline
    with st.spinner("🤖 AI is analyzing restaurants..."):
        try:
            response = get_recommendations(df, validated)
        except Exception as e:
            render_error(str(e))
            logger.exception("Recommendation pipeline failed")
            return

    result = response.get("result")

    if result is None or not result.recommendations:
        render_no_results(validated.location)
        return

    # Store in session state for persistence across reruns
    st.session_state["last_result"] = result
    st.session_state["last_metadata"] = response

    _display_results(result, response)


def _display_results(result, metadata: dict):
    """Display the recommendation results."""
    # Summary card
    render_summary(result, metadata)

    # Recommendation cards
    render_recommendations(result)

    # Cached indicator
    if metadata.get("cached"):
        st.caption("⚡ Served from cache")


# ===================================================================
# Entry Point
# ===================================================================
if __name__ == "__main__":
    main()
