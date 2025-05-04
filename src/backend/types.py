from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchInputs:
    description_query: str
    category_query: Optional[str] = None
    calories_val: Optional[int] = None

@dataclass
class WeightedParams:
    desc_weight: float = 1.0
    cat_weight: float = 1.0

@dataclass
class NumericParams:
    desc_weight: float = 1.0
    cal_weight: float = 1.0

@dataclass
class CombinedParams:
    desc_weight: float = 1.0
    cal_weight: float = 1.0
