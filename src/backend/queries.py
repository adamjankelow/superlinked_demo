"""
Query functions with shared SearchCtx and structured SearchInputs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
from superlinked import framework as sl


# ───────────────────────── context & inputs ─────────────────────────────
@dataclass(frozen=True)
class SearchCtx:
    app: object
    index: object
    food_item: object
    desc_space: object
    cat_text_space: object
    cat_cat_space: object
    cal_space: object

# ───────────────────────── weights (optional) ───────────────────────────
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


_COLS = ["description", "food_category", "calories", "similarity_score"]


# ───────────────────────── search functions ─────────────────────────────
def simple_search(ctx: SearchCtx, inp: SearchInputs) -> pd.DataFrame:
    q = (
        sl.Query(ctx.index)
        .find(ctx.food_item)
        .similar(ctx.desc_space, sl.Param("q"))
        .select_all()
    )
    res = ctx.app.query(q, q=inp.description_query)
    return sl.PandasConverter.to_pandas(res)[_COLS]


def weighted_search(ctx: SearchCtx, inp: SearchInputs, p: WeightedParams) -> pd.DataFrame:
    q = (
        sl.Query(
            ctx.index,
            weights={
                ctx.desc_space: p.desc_weight,
                ctx.cat_text_space: p.cat_weight,
                ctx.cat_cat_space:  p.cat_weight,
            },
        )
        .find(ctx.food_item)
        .similar(ctx.desc_space, sl.Param("q"))
        .similar(ctx.cat_text_space, sl.Param("cat"))
        .similar(ctx.cat_cat_space,  sl.Param("cat"))
        .select_all()
    )
    res = ctx.app.query(q, q=inp.description_query, cat=inp.category_query)
    return sl.PandasConverter.to_pandas(res)[_COLS]


def numeric_search(ctx: SearchCtx, inp: SearchInputs, p: NumericParams) -> Tuple[pd.DataFrame, float]:
    q = (
        sl.Query(
            ctx.index,
            weights={ctx.desc_space: p.desc_weight, ctx.cal_space: p.cal_weight},
        )
        .find(ctx.food_item)
        .similar(ctx.desc_space, sl.Param("q"))
        .similar(ctx.cal_space, sl.Param("cal"))
        .select_all()
    )
    res = ctx.app.query(q, q=inp.description_query, cal=inp.calories_val)
    df = sl.PandasConverter.to_pandas(res)[_COLS]
    top10 = df.head(10)
    return top10, top10["calories"].mean() if not top10.empty else 0.0


def combined_search(ctx: SearchCtx, inp: SearchInputs, p: CombinedParams) -> pd.DataFrame:
    q = (
        sl.Query(
            ctx.index,
            weights={ctx.desc_space: p.desc_weight, ctx.cal_space: p.cal_weight},
        )
        .find(ctx.food_item)
        .similar(ctx.cat_cat_space.category, sl.Param("cat"))
        .similar(ctx.desc_space, sl.Param("q"))
        .similar(ctx.cal_space, sl.Param("cal"))
        .select_all()
    )
    res = ctx.app.query(
        q,
        q=inp.description_query,
        cat=inp.category_query,
        cal=inp.calories_val,
    )
    return sl.PandasConverter.to_pandas(res)[_COLS]
