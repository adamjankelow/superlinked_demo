from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SearchCtx:
    app: object
    index: object
    food_item: object
    desc_space: object
    cat_text_space: object
    cat_cat_space: object
    cal_space: object

@dataclass
class SearchInputs:
    description_query: str
    category_query: Optional[str] = None
    calories_val: Optional[int] = None

@dataclass
@dataclass
class SearchWeights:
    desc_weight: float = 1.0
    cat_weight: float = 1.0
    cal_weight: float = 1.0

