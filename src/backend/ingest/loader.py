"""
Load the food database and build the Superlinked app.
"""

import pandas as pd
from pathlib import Path
from joblib import Memory
from superlinked import framework as sl
from ..config import settings
from .schema import FoodItem

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



memory = Memory(settings.cache_dir, verbose=0)

@memory.cache
def build_superlinked_app(df):
    """
    Build and configure the Superlinked application for semantic search.

    Args:
        df (pd.DataFrame): The DataFrame containing the food database.

    Returns:
        tuple: A tuple containing the app, index, food_item, and various spaces.
    """
    food_item = FoodItem()
    categories = df["food_category"].unique().tolist()
    # # Spaces
    description_space = sl.TextSimilaritySpace(text=food_item.description, model=settings.embedding_model)
    # Semantic similarity over food category text
    food_category_text_space = sl.TextSimilaritySpace(text=food_item.food_category, model=settings.embedding_model)

    # Exact/category-level similarity (discrete match)
    food_category_categorical_space = sl.CategoricalSimilaritySpace(
        category_input=food_item.food_category,
        categories=categories
    )
    calories_space = sl.NumberSpace(
        food_item.calories,
        min_value=settings.calories_min,
        max_value=settings.calories_max,
        mode=sl.Mode.SIMILAR
    )

    index = sl.Index([description_space, food_category_text_space, food_category_categorical_space, calories_space])

    # Set up engine
    source = sl.InMemorySource(food_item)
    executor = sl.InMemoryExecutor(sources=[source], indices=[index])
    app = executor.run()

    # Insert data
    source.put(
        df[["fdc_id", "description", "food_category", "calories"]].to_dict(orient="records")
    )

    return app, index, food_item, description_space, food_category_text_space, food_category_categorical_space, calories_space


