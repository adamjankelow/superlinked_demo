"""
Load the food database and build the Superlinked app.
"""

import pandas as pd
from pathlib import Path
from superlinked import framework as sl
from ..config import settings
from .schema import FoodItem
from ..search.types import SearchCtx

# ---- Load Data ----

def load_data():
    """
    Load the food database from a Parquet file.

    Returns:
        pd.DataFrame: A DataFrame containing the food database.
    """

    repo_root = Path(__file__).resolve().parents[3]
    data_file = repo_root / settings.data_path
    df = pd.read_parquet(data_file)
    return df



def build_superlinked_app(df):
    """
    Constructs and returns a fully-ingested Superlinked SearchCtx object.

    This function initializes the schema for food items and sets up various
    similarity spaces for text, categorical, and numerical data. It builds
    an index using these spaces and ingests the provided DataFrame into an
    in-memory source. The function then executes the Superlinked application
    and returns a SearchCtx object containing the application context, index,
    and similarity spaces.

    Args:
        df (pd.DataFrame): A DataFrame containing food data with columns
                           'fdc_id', 'description', 'food_category', and 'calories'.

    Returns:
        SearchCtx: An object containing the Superlinked application context,
                   index, and similarity spaces for further search operations.
    """
    schema = FoodItem()
    categories = df.food_category.unique().tolist()

    desc_space  = sl.TextSimilaritySpace(text=schema.description, model=settings.embedding_model)
    cat_text    = sl.TextSimilaritySpace(text=schema.food_category, model=settings.embedding_model)
    cat_cat     = sl.CategoricalSimilaritySpace(category_input=schema.food_category, categories=categories)
    cal_space   = sl.NumberSpace(schema.calories,
                                 min_value=settings.calories_min,
                                 max_value=settings.calories_max,
                                 mode=sl.Mode.SIMILAR)

    index = sl.Index([desc_space, cat_text, cat_cat, cal_space])
    source = sl.InMemorySource(schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[index])
    app = executor.run()

    records = df[["fdc_id", "description", "food_category", "calories"]].to_dict(orient="records")
    source.put(records)

    return SearchCtx(
        app=app,
        index=index,
        food_item=schema,
        desc_space=desc_space,
        cat_text_space=cat_text,
        cat_cat_space=cat_cat,
        cal_space=cal_space,
    )

