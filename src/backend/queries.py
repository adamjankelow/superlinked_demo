"""
Query functions for semantic search using Superlinked.
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd
from superlinked import framework as sl
from .types import SearchInputs, WeightedParams, NumericParams, CombinedParams, SearchCtx



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
