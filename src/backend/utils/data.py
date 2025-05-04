import pandas as pd
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from superlinked import framework as sl
# from huggingface_hub import snapshot_download
from ..config import settings

# ---- Load Data ----

def load_data():
    # data.py is at <repo>/src/backend/utils/data.py
    repo_root = Path(__file__).resolve().parents[3]
    data_file = repo_root / settings.data_path
    df = pd.read_parquet(data_file)
    return df



# ---- Define schema ----
class FoodItem(sl.Schema):
    fdc_id : sl.IdField
    description : sl.String
    food_category : sl.String
    calories : sl.Float


# ---- Build Superlinked App ----
def build_superlinked_app(df):
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


def get_umap_df() -> pd.DataFrame:
    """
    Returns a DataFrame with `umap_x`, `umap_y` + the original metadata.
    """

    repo_root = Path(__file__).resolve().parents[3]
    umap_file = repo_root / settings.umap_path
    df = pd.read_parquet(umap_file)
    return df



