"""
User preferences data model.

Defines the UserPreferences dataclass used throughout the pipeline
to represent what the user is looking for in a restaurant.
"""

from dataclasses import dataclass, field

# Valid values for constrained fields
VALID_BUDGETS = ("low", "medium", "high")
VALID_PREFERENCES = (
    "family-friendly",
    "quick service",
    "romantic",
    "outdoor seating",
    "live music",
    "pet-friendly",
    "vegan options",
    "vegetarian",
    "late night",
    "breakfast",
    "buffet",
    "fine dining",
    "casual dining",
    "street food",
)


@dataclass
class UserPreferences:
    """
    Represents user preferences for restaurant recommendations.

    Attributes:
        location: City or area name (e.g., "Delhi", "Bangalore").
        budget: Budget tier — "low", "medium", or "high".
        cuisine: Preferred cuisine type (e.g., "Italian", "Chinese").
                 None means no cuisine preference.
        min_rating: Minimum acceptable rating (0.0 – 5.0).
        preferences: Additional preference tags (e.g., ["family-friendly"]).
    """

    location: str
    budget: int | None = None
    party_size: int = 2
    cuisine: str | None = None
    min_rating: float = 3.5
    preferences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        return {
            "location": self.location,
            "budget": self.budget,
            "party_size": self.party_size,
            "cuisine": self.cuisine,
            "min_rating": self.min_rating,
            "preferences": self.preferences,
        }

    def to_display_string(self) -> str:
        """Return a human-readable summary of preferences."""
        parts = [f"Location: {self.location}"]
        if self.budget:
            parts.append(f"Budget: ₹{self.budget} for {self.party_size} people")
        if self.cuisine:
            parts.append(f"Cuisine: {self.cuisine}")
        parts.append(f"Min Rating: {self.min_rating}")
        if self.preferences:
            parts.append(f"Preferences: {', '.join(self.preferences)}")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.to_display_string()
