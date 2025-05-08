"""
Types for semantic search using Superlinked.
"""

from dataclasses import dataclass
from typing import Optional
import superlinked as sl

@dataclass(frozen=True)
class SearchCtx:
    """
    Represents the context for a search operation using Superlinked.

    Attributes:
        app: The Superlinked application instance.
        index: The index used for searching.
        food_item: The schema for the food item.
        desc_space: The text similarity space for descriptions.
        cat_text_space: The text similarity space for categories.
        cat_cat_space: The categorical similarity space for categories.
        cal_space: The numerical similarity space for calories.
    """
    app: object
    index: object
    food_item: object
    desc_space: object
    cat_text_space: object
    cat_cat_space: object
    cal_space: object


@dataclass
class SearchInputs:
    """
    Represents the inputs for a search operation.

    Attributes:
        description_query: The query string for the description.
        category_query: An optional query string for the category.
        calories_val: An optional integer value for calories.
    """
    description_query: str
    category_query: Optional[str] = None
    calories_val: Optional[int] = None

@dataclass
class CategoryWeights:
    """
    Represents the weights for category-based search.

    Attributes:
        desc_weight: The weight for the description similarity.
        cat_weight: The weight for the category similarity.
    """
    desc_weight: float = 1.0
    cat_weight:  float = 1.0

@dataclass
class NumericWeights:
    """
    Represents the weights for numeric-based search.

    Attributes:
        desc_weight: The weight for the description similarity.
        cal_weight: The weight for the calorie similarity.
    """
    desc_weight: float = 1.0
    cal_weight:  float = 1.0
