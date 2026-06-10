"""
Preference collector module.

Provides a programmatic API for collecting and building UserPreferences
objects, suitable for use by the Streamlit UI or CLI interfaces.
"""

import logging
from typing import Optional

import pandas as pd

from src.data.preprocessing import get_unique_locations, get_unique_cuisines
from src.data.schema import CleanColumns
from src.input.models import UserPreferences, VALID_BUDGETS, VALID_PREFERENCES
from src.input.validator import validate_and_normalize

logger = logging.getLogger("zomato_recommender.input.collector")


class PreferenceCollector:
    """
    Collects and validates user preferences against the loaded dataset.

    This class acts as the bridge between the UI layer and the
    validation/filtering pipeline. It knows about available locations
    and cuisines from the dataset and uses them for validation.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the collector with the preprocessed dataset.

        Args:
            df: Preprocessed DataFrame to derive valid options from.
        """
        self._df = df
        self._locations = get_unique_locations(df)
        self._cuisines = get_unique_cuisines(df)

        logger.info(
            f"PreferenceCollector initialized: "
            f"{len(self._locations)} locations, {len(self._cuisines)} cuisines"
        )

    @property
    def available_locations(self) -> list[str]:
        """Return sorted list of available locations."""
        return self._locations

    @property
    def available_cuisines(self) -> list[str]:
        """Return sorted list of available cuisines."""
        return self._cuisines

    @property
    def available_budgets(self) -> tuple:
        """Return valid budget options."""
        return VALID_BUDGETS

    @property
    def available_preferences(self) -> tuple:
        """Return valid additional preference tags."""
        return VALID_PREFERENCES

    def collect(
        self,
        location: str,
        budget: str = "medium",
        cuisine: Optional[str] = None,
        min_rating: float = 3.5,
        preferences: Optional[list[str]] = None,
    ) -> tuple[UserPreferences, list[str]]:
        """
        Validate inputs and return a UserPreferences object.

        Args:
            location: City or area name.
            budget: Budget tier.
            cuisine: Preferred cuisine (optional).
            min_rating: Minimum acceptable rating.
            preferences: Additional preference tags.

        Returns:
            Tuple of (UserPreferences, list of warning messages).

        Raises:
            ValueError: If location is missing or critically invalid.
        """
        return validate_and_normalize(
            location=location,
            budget=budget,
            cuisine=cuisine,
            min_rating=min_rating,
            preferences=preferences,
            known_locations=self._locations,
            known_cuisines=self._cuisines,
        )

    def get_location_stats(self) -> dict[str, int]:
        """Return restaurant count per location."""
        if CleanColumns.LOCATION not in self._df.columns:
            return {}
        return self._df[CleanColumns.LOCATION].value_counts().to_dict()

    def get_cuisine_stats(self) -> dict[str, int]:
        """Return restaurant count per cuisine (from cuisines_list)."""
        if CleanColumns.CUISINES_LIST not in self._df.columns:
            return {}

        cuisine_counts: dict[str, int] = {}
        for cuisines in self._df[CleanColumns.CUISINES_LIST].dropna():
            if isinstance(cuisines, list):
                for c in cuisines:
                    cuisine_counts[c] = cuisine_counts.get(c, 0) + 1

        return dict(sorted(cuisine_counts.items(), key=lambda x: -x[1]))
