import pandas as pd
import os
import streamlit as st
from superlinked import framework as sl

# ---- Load Data ----
@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    
    # Construct the absolute path to the dataset
    dataset_path = os.path.join(current_dir, "../data/sr_legacy_food_db.parquet")
    absolute_path = os.path.abspath(dataset_path)
    df = pd.read_parquet(absolute_path)  # replace with your dataset path
    df = df.dropna(subset=["description", "food_category", "Energy"]).rename(columns={"Energy": "calories"})
    df = df.sample(n=200)  # optional: limit for speed
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

    # Spaces
    description_space = sl.TextSimilaritySpace(text=food_item.description, model="all-MiniLM-L6-v2")
    # Semantic similarity over food category text
    food_category_text_space = sl.TextSimilaritySpace(text=food_item.food_category, model="all-MiniLM-L6-v2")

    # Exact/category-level similarity (discrete match)
    food_category_categorical_space = sl.CategoricalSimilaritySpace(
        category_input=food_item.food_category,
        categories=categories
    )

    index = sl.Index([description_space, food_category_text_space, food_category_categorical_space])

    # Set up engine
    source = sl.InMemorySource(food_item)
    executor = sl.InMemoryExecutor(sources=[source], indices=[index])
    app = executor.run()

    # Insert data
    source.put(
        df[["fdc_id", "description", "food_category", "calories"]].to_dict(orient="records")
    )

    return app, index, food_item, description_space, food_category_text_space, food_category_categorical_space