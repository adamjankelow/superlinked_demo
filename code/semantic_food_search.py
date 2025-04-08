import streamlit as st
import pandas as pd
from superlinked import framework as sl
import sys
from queries import simple_search, weighted_search, numeric_search, combined_search
from utills import load_data, build_superlinked_app, create_umap_df, plot_umap_scatter

"""
import os 

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

"""
# ---- Streamlit UI ----
st.title("🥦 Semantic Search on Food Database")


# Sidebar for additional options
# st.sidebar.header("Search Options")


# Load data and build app
df = load_data()
app, index, food_item, description_space, food_category_text_space, food_category_categorical_space, calories_space = build_superlinked_app(df)


mode = st.sidebar.radio(
    "Select Search Mode",
    ["Simple Search", "Weighted Search", "Numeric Search", "Combined Search"]
)

if mode == "Simple Search":
    food_item_df = simple_search(food_item, description_space, index, app)
    
                
if mode == "Weighted Search":
    food_item_df = weighted_search(food_item, description_space, food_category_text_space, food_category_categorical_space, index, app)
    
    
if mode == "Numeric Search":  
    numeric_search(food_item, description_space, calories_space, index, app)
 
if mode == "Combined Search":
    categories = df.food_category.drop_duplicates().to_list()
    combined_search(food_item, description_space, food_category_categorical_space, calories_space, index, app, categories)

                
# ---- Run the app ----
# streamlit run semantic_food_search.py