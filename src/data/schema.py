"""
Data schema definitions and column constants.

Centralizes all column name references, budget tier thresholds,
and data model definitions to ensure consistency across the pipeline.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Raw Column Names (as they appear in the Hugging Face dataset)
# ---------------------------------------------------------------------------
class RawColumns:
    """Column names from the raw Zomato dataset."""

    RESTAURANT_NAME = "name"
    ONLINE_ORDER = "online_order"
    BOOK_TABLE = "book_table"
    RATING = "rate"
    VOTES = "votes"
    PHONE = "phone"
    LOCATION = "location"
    REST_TYPE = "rest_type"
    DISH_LIKED = "dish_liked"
    CUISINES = "cuisines"
    COST_FOR_TWO = "approx_cost(for two people)"
    REVIEWS_LIST = "reviews_list"
    MENU_ITEM = "menu_item"
    LISTED_IN_TYPE = "listed_in(type)"
    LISTED_IN_CITY = "listed_in(city)"


# ---------------------------------------------------------------------------
# Cleaned Column Names (after preprocessing)
# ---------------------------------------------------------------------------
class CleanColumns:
    """Standardized column names used after preprocessing."""

    RESTAURANT_NAME = "restaurant_name"
    ONLINE_ORDER = "online_order"
    BOOK_TABLE = "book_table"
    RATING = "rating"
    VOTES = "votes"
    LOCATION = "location"
    RESTAURANT_TYPE = "restaurant_type"
    CUISINES = "cuisines"
    CUISINES_LIST = "cuisines_list"
    COST_FOR_TWO = "cost_for_two"
    BUDGET_TIER = "budget_tier"
    LISTED_IN_TYPE = "listed_in_type"
    LISTED_IN_CITY = "listed_in_city"


# ---------------------------------------------------------------------------
# Budget Tier Configuration
# ---------------------------------------------------------------------------
BUDGET_TIERS = {
    "low": (0, 400),        # ₹0 – ₹400 (inclusive upper)
    "medium": (401, 1000),  # ₹401 – ₹1000
    "high": (1001, float("inf")),  # ₹1001+
}

BUDGET_TIER_ORDER = ["low", "medium", "high"]


def get_budget_tier(cost: float | None) -> str | None:
    """
    Map a cost-for-two value to a budget tier.

    Args:
        cost: The cost for two people in ₹.

    Returns:
        Budget tier string ('low', 'medium', 'high') or None if cost is invalid.
    """
    if cost is None or cost < 0:
        return None

    for tier, (low, high) in BUDGET_TIERS.items():
        if low <= cost <= high:
            return tier

    return "high"  # Fallback for very high costs


# ---------------------------------------------------------------------------
# Required Columns (must be present after preprocessing)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    CleanColumns.RESTAURANT_NAME,
    CleanColumns.LOCATION,
    CleanColumns.RATING,
    CleanColumns.COST_FOR_TWO,
]

# Columns that are nice to have but not strictly required
OPTIONAL_COLUMNS = [
    CleanColumns.VOTES,
    CleanColumns.CUISINES,
    CleanColumns.CUISINES_LIST,
    CleanColumns.ONLINE_ORDER,
    CleanColumns.BOOK_TABLE,
    CleanColumns.RESTAURANT_TYPE,
    CleanColumns.BUDGET_TIER,
]


# ---------------------------------------------------------------------------
# Data Model for a cleaned restaurant record
# ---------------------------------------------------------------------------
@dataclass
class Restaurant:
    """Represents a single cleaned restaurant record."""

    restaurant_name: str
    location: str
    rating: float
    cost_for_two: int
    cuisines: str = ""
    cuisines_list: list[str] | None = None
    votes: int = 0
    online_order: bool = False
    book_table: bool = False
    restaurant_type: str = ""
    budget_tier: str = "medium"

    def matches_cuisine(self, cuisine: str) -> bool:
        """Check if this restaurant serves the given cuisine."""
        if not self.cuisines_list:
            return False
        return cuisine.lower().strip() in [c.lower() for c in self.cuisines_list]

    def to_display_dict(self) -> dict:
        """Convert to a dictionary suitable for display."""
        return {
            "Restaurant Name": self.restaurant_name,
            "Location": self.location,
            "Cuisine": self.cuisines,
            "Rating": f"{self.rating}/5",
            "Cost for Two": f"₹{self.cost_for_two}",
            "Votes": self.votes,
            "Online Order": "Yes" if self.online_order else "No",
            "Book Table": "Yes" if self.book_table else "No",
        }
