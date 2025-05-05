"""
Query functions for semantic search using Superlinked.
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd
from superlinked import framework as sl
from .types import SearchInputs, SearchWeights, SearchCtx



_COLS = ["description", "food_category", "calories", "similarity_score"]


# ───────────────────────── search functions ─────────────────────────────
def simple_search(ctx: SearchCtx, inp: SearchInputs) -> pd.DataFrame:
    """
    Perform a simple search based on the description query.

    Args:
        ctx (SearchCtx): The search context containing the index and spaces.
        inp (SearchInputs): The search inputs containing the description query.

    Returns:
        pd.DataFrame: A DataFrame containing the search results with specified columns.
    """
    q = (
        sl.Query(ctx.index)
        .find(ctx.food_item)
        .similar(ctx.desc_space, sl.Param("q"))
        .select_all()
    )
    res = ctx.app.query(q, q=inp.description_query)
    return sl.PandasConverter.to_pandas(res)[_COLS]


def weighted_search(ctx: SearchCtx, inp: SearchInputs, p: SearchWeights) -> pd.DataFrame:
    """
    Perform a weighted search based on description and category queries.

    Args:
        ctx (SearchCtx): The search context containing the index and spaces.
        inp (SearchInputs): The search inputs containing the description and category queries.
        p (SearchWeights): The parameters containing weights for description and category spaces.

    Returns:
        pd.DataFrame: A DataFrame containing the search results with specified columns.
    """
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


def numeric_search(ctx: SearchCtx, inp: SearchInputs, p: SearchWeights) -> Tuple[pd.DataFrame, float]:
    """
    Perform a numeric search based on description and calories queries.

    Args:
        ctx (SearchCtx): The search context containing the index and spaces.
        inp (SearchInputs): The search inputs containing the description and calories queries.
        p (SearchWeights): The parameters containing weights for description and calories spaces.

    Returns:
        Tuple[pd.DataFrame, float]: A tuple containing a DataFrame with the top 10 search results
                                    and the mean calories of these results.
    """
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


def combined_search(ctx: SearchCtx, inp: SearchInputs, p: SearchWeights) -> pd.DataFrame:
    """
    Perform a combined search based on category, description, and calories queries.

    Args:
        ctx (SearchCtx): The search context containing the index and spaces.
        inp (SearchInputs): The search inputs containing the category, description, and calories queries.
        p (SearchWeights): The parameters containing weights for description and calories spaces.

    Returns:
        pd.DataFrame: A DataFrame containing the search results with specified columns.
    """
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

