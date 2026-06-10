"""
Renderer module — builds all Streamlit UI components.

Provides functions for rendering:
- App header
- Sidebar preference inputs
- AI summary card
- Recommendation cards
- Stats chips
- Empty / error states
- Footer
"""

import streamlit as st
import pandas as pd

from src.data.preprocessing import get_unique_locations, get_unique_cuisines
from src.input.models import VALID_BUDGETS, VALID_PREFERENCES
from src.recommendation.parser import RecommendationResult


# ===================================================================
# Header
# ===================================================================
def render_header():
    """Render the app header with title and subtitle."""
    st.markdown(
        """
        <div class="app-header">
            <h1>🍽️ Zomato AI Recommender</h1>
            <p>AI-powered restaurant recommendations, just for you</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================================================================
# Sidebar
# ===================================================================
def render_sidebar(df: pd.DataFrame) -> dict:
    """
    Render the sidebar with preference inputs.

    Args:
        df: Preprocessed DataFrame to populate dropdowns.

    Returns:
        Dictionary with user selections.
    """
    with st.sidebar:
        st.markdown('<p class="sidebar-header">🎯 Your Preferences</p>', unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        with st.form("preferences_form"):
            # Location
            locations = get_unique_locations(df)
            location = st.selectbox(
                "📍 Location",
                options=locations,
                index=0 if locations else None,
                help="Select the area where you want to dine",
            )

            # Budget (Slider - no +/- buttons)
            budget = st.slider(
                "💸 Total Budget (₹)",
                min_value=100,
                max_value=3000,
                value=1000,
                step=50,
                help="Your total budget in Rupees",
            )
            
            # Party Size (Slider - no +/- buttons)
            party_size = st.slider(
                "👥 Number of People (optional)",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                help="How many people are going?",
            )

            # Cuisine
            cuisines = get_unique_cuisines(df)
            cuisine_options = ["Any (no preference)"] + cuisines
            cuisine_selection = st.selectbox(
                "🍕 Cuisine",
                options=cuisine_options,
                index=0,
                help="Select your preferred cuisine type",
            )
            cuisine = None if cuisine_selection == "Any (no preference)" else cuisine_selection

            # Min Rating (Smooth 0.1 steps)
            min_rating = st.slider(
                "⭐ Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                value=3.5,
                step=0.1,
                help="Only show restaurants rated at or above this",
            )

            # Additional Preferences
            st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
            preferences = st.multiselect(
                "🏷️ Additional Preferences",
                options=list(VALID_PREFERENCES),
                default=[],
                help="Select any additional dining preferences",
            )

            st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

            # Submit button
            submitted = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)

    return {
        "location": location,
        "budget": budget,
        "party_size": party_size,
        "cuisine": cuisine,
        "min_rating": min_rating,
        "preferences": preferences,
        "submitted": submitted,
    }


# ===================================================================
# Summary Card
# ===================================================================
def render_summary(result: RecommendationResult, metadata: dict):
    """Render the AI summary card above recommendations."""
    summary_text = result.summary or "Here are the best restaurants matching your criteria."
    model_used = result.model_used or "unknown"
    candidates = metadata.get("candidates_found", 0)
    response_time = metadata.get("response_time_ms", 0)

    # Filters info
    relaxations = metadata.get("metadata", {}).get("relaxations", [])

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-icon">🤖 AI Summary</div>
            <div class="summary-text">{summary_text}</div>
            <div class="summary-meta">
                Model: {model_used} &nbsp;•&nbsp;
                {candidates} candidates found &nbsp;•&nbsp;
                {response_time}ms response time
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats chips
    chips_html = '<div class="stats-row">'
    chips_html += f'<span class="stat-chip">🎯 <span class="stat-value">{len(result.recommendations)}</span> recommendations</span>'
    chips_html += f'<span class="stat-chip">📊 <span class="stat-value">{candidates}</span> candidates screened</span>'

    if relaxations:
        chips_html += f'<span class="stat-chip">🔄 <span class="stat-value">{len(relaxations)}</span> filter relaxation(s)</span>'

    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


# ===================================================================
# Recommendation Cards
# ===================================================================
def render_recommendations(result: RecommendationResult):
    """Render all recommendation cards."""
    for rec in result.recommendations:
        _render_single_card(rec)


def _render_single_card(rec):
    """Render a single recommendation card."""
    rank = rec.rank
    rank_class = f"rank-{rank}" if rank <= 3 else "rank-default"
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

    # Use HTML for the beautiful outer card and header
    card_html = f"""
        <div class="rec-card">
            <div class="rec-header-container">
                <div class="rank-badge {rank_class}">{rank_emoji}</div>
                <h3 class="rec-title">{rec.restaurant_name}</h3>
            </div>
            <div style="display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap;">
                <div class="metric-badge">🍕 {rec.cuisine}</div>
                <div class="metric-badge"><span class="rating-badge">⭐ {rec.rating:.1f}/5</span></div>
                <div class="metric-badge">💰 {rec.estimated_cost}</div>
            </div>
        </div>
    """
    
    # We use a container to group the HTML card and the interactive expander
    with st.container():
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Interactive Expander for AI Explanation (attached below the header)
        with st.expander("💡 Read AI's detailed recommendation..."):
            st.write(rec.explanation)
            st.markdown("---")
            
            # Interactive Buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🍽️ Book Table", key=f"book_{rank}_{rec.restaurant_name}"):
                    st.toast(f"Redirecting to Zomato booking for {rec.restaurant_name}...", icon="🍽️")
            with col2:
                if st.button("📍 Directions", key=f"dir_{rank}_{rec.restaurant_name}"):
                    st.toast(f"Opening maps for {rec.restaurant_name}...", icon="📍")


# ===================================================================
# Warnings
# ===================================================================
def render_warnings(warnings: list[str]):
    """Render any validation warnings."""
    for w in warnings:
        st.warning(w, icon="⚠️")


# ===================================================================
# Empty / Error States
# ===================================================================
def render_empty_state():
    """Render the initial empty state before any search."""
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🍽️</div>
            <h3>Ready to find your perfect restaurant?</h3>
            <p style="color: #aaa; margin-top: 10px;">
                Select your preferences in the sidebar and click
                <strong>"Get Recommendations"</strong> to discover
                AI-curated dining suggestions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_no_results(location: str):
    """Render the no-results state."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size: 3rem; margin-bottom: 1rem;">😕</div>
            <h3>No restaurants found</h3>
            <p style="color: #aaa; margin-top: 10px;">
                We couldn't find restaurants matching your criteria in
                <strong>{location}</strong>. Try relaxing your filters —
                broaden the cuisine, lower the rating, or change the budget.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(error_message: str):
    """Render an error state."""
    st.error(f"Something went wrong: {error_message}", icon="❌")
    st.info(
        "Try again in a moment. If the issue persists, check your API key in `.env`.",
        icon="💡",
    )


# ===================================================================
# Footer
# ===================================================================
def render_footer():
    """Render the app footer."""
    st.markdown(
        """
        <div class="app-footer">
            Powered by <strong>Groq AI</strong> (Llama 3.3) &nbsp;•&nbsp;
            Data: Zomato (Hugging Face) &nbsp;•&nbsp;
            Built with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
