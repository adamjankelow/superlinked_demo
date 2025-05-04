# backend/queries.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
from superlinked import framework as sl

_COLS = ["description", "food_category", "calories", "similarity_score"]


# --------------------------- parameter packs --------------------------- #
@dataclass
class WeightedParams:
    desc_weight: float = 1.0
    cat_weight: float = 1.0


@dataclass
class NumericParams:
    desc_weight: float = 1.0
    calories_weight: float = 1.0


# --------------------------- simple search --------------------------- #
def simple_search(
    app,
    index,
    food_item,
    desc_space,
    query_text: str,
) -> pd.DataFrame:
    q = (
        sl.Query(index)
        .find(food_item)
        .similar(desc_space, sl.Param("query_text"))
        .select_all()
    )
    res = app.query(q, query_text=query_text)
    return sl.PandasConverter.to_pandas(res)[_COLS]


# --------------------------- weighted search --------------------------- #
def weighted_search(
    app,
    index,
    food_item,
    desc_space,
    cat_text_space,
    cat_cat_space,
    query_text: str,
    food_category: str,
    p: WeightedParams,
) -> pd.DataFrame:
    q = (
        sl.Query(
            index,
            weights={
                desc_space: p.desc_weight,
                cat_text_space: p.cat_weight,
                cat_cat_space: p.cat_weight,
            },
        )
        .find(food_item)
        .similar(desc_space, sl.Param("query_text"))
        .similar(cat_text_space, sl.Param("food_category"))
        .similar(cat_cat_space, sl.Param("food_category"))
        .select_all()
    )
    res = app.query(q, query_text=query_text, food_category=food_category)
    return sl.PandasConverter.to_pandas(res)


# --------------------------- numeric search --------------------------- #
def numeric_search(
    app,
    index,
    food_item,
    desc_space,
    cal_space,
    query_text: str,
    calories_val: int,
    p: NumericParams,
) -> Tuple[pd.DataFrame, float]:
    q = (
        sl.Query(
            index,
            weights={desc_space: p.desc_weight, cal_space: p.calories_weight},
        )
        .find(food_item)
        .similar(desc_space, sl.Param("query_text"))
        .similar(cal_space, sl.Param("calories_val"))
        .select_all()
    )
    res = app.query(q, query_text=query_text, calories_val=calories_val)
    df = sl.PandasConverter.to_pandas(res)[_COLS]
    top10 = df.head(10)
    return top10, (top10["calories"].mean() if not top10.empty else 0.0)


# --------------------------- combined search --------------------------- #
@dataclass
class CombinedParams:
    desc_weight: float = 1.0
    calories_weight: float = 1.0


def combined_search(
    app,
    index,
    food_item,
    desc_space,
    cat_cat_space,
    cal_space,
    query_text: str,
    category_filter: str,
    calories_val: int,
    p: CombinedParams,
) -> pd.DataFrame:
    """
    Hard‑filters by `category_filter`, then applies text+calorie weighting.
    """
    q = (
        sl.Query(
            index,
            weights={desc_space: p.desc_weight, cal_space: p.calories_weight},
        )
        .find(food_item)
        .similar(cat_cat_space.category, sl.Param("category_filter"))
        .similar(desc_space, sl.Param("query_text"))
        .similar(cal_space, sl.Param("calories_val"))
        .select_all()
    )
    res = app.query(
        q,
        category_filter=category_filter,
        query_text=query_text,
        calories_val=calories_val,
    )
    return sl.PandasConverter.to_pandas(res)[_COLS]
