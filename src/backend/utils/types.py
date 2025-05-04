from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchInputs:
    description_query: str
    category_query: Optional[str] = None
    calories_val: Optional[int] = None