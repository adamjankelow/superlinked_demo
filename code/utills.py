import pandas as pd
import os
import streamlit as st
from superlinked import framework as sl
import numpy as np


# ---- Load Data ----
@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    
    # Construct the absolute path to the dataset
    dataset_path = os.path.join(current_dir, "../data/sr_legacy_food_db.parquet")
    absolute_path = os.path.abspath(dataset_path)
    df = pd.read_parquet(absolute_path)  # replace with your dataset path
    df = df.dropna(subset=["description", "food_category", "Energy"]).rename(columns={"Energy": "calories"})
    return df.sample(n=200)



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
    energy_space = sl.NumberSpace(
        food_item.calories,
        min_value=0,
        max_value=1000,
        mode=sl.Mode.SIMILAR
)

    index = sl.Index([description_space, food_category_text_space, food_category_categorical_space, energy_space])

    # Set up engine
    source = sl.InMemorySource(food_item)
    executor = sl.InMemoryExecutor(sources=[source], indices=[index])
    app = executor.run()

    # Insert data
    source.put(
        df[["fdc_id", "description", "food_category", "calories"]].to_dict(orient="records")
    )

    return app, index, food_item, description_space, food_category_text_space, food_category_categorical_space, energy_space


#queries
def simple_search(food_item, description_space, index, app):
    st.markdown("Use this app to search for food items based on their descriptions. Enter a query below to get started.")
    query_input = st.text_input("Search for a food", "sugary cereal")
    if query_input:
        with st.spinner('Searching...'):
            try:
                query = (
                    sl.Query(index)
                .find(food_item)
                .similar(description_space, sl.Param("query_text"))
                .select_all()
                )

                result = app.query(query, query_text=query_input)
                df_result = sl.PandasConverter.to_pandas(result)[["description", "food_category", "calories", 'similarity_score']]
                st.write(f"🔍 Top results for: **{query_input}**")
                st.dataframe(df_result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
def weighted_search(food_item, description_space, food_category_text_space, food_category_categorical_space, index, app):
    st.markdown("Superlinked supports weighted search. Use the sliders below to adjust the weights of the description and category search spaces to see how they affect the results.")
    query = st.text_input("Search for a food")
    food_category = st.text_input("Search for a food category")
    desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
    cat_weight = st.slider("Category weight", -3.0, 3.0, 1.0)

    if query:
        q = (
            sl.Query(index, weights={description_space: sl.Param("desc_weight"), food_category_text_space: sl.Param("cat_weight"), food_category_categorical_space: sl.Param("cat_weight")})
            .find(food_item)
            .similar(description_space, sl.Param("food_item"))
            .similar(food_category_text_space, sl.Param("food_category"))
            .similar(food_category_categorical_space, sl.Param("query_text"))
            .select_all()
        )
        result = app.query(q, food_item=query, food_category=food_category, desc_weight=desc_weight, cat_weight=cat_weight)
        st.dataframe(sl.PandasConverter.to_pandas(result))
        
def numeric_search(food_item, description_space, energy_space, index, app):
    st.markdown("We can also include numeric spaces in our search. See how the results change as we change the weights of the description and calories.")
    desc_input = st.text_input("Search for a food")
    energy_input = st.number_input("Search for a calorie value per 100g", min_value=0, max_value=1000)
    desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
    energy_weight = st.slider("Energy weight", -3.0, 3.0, 1.0)
    if desc_input and energy_input:
        with st.spinner('Searching...'):
            try:
                query = (
                    sl.Query(index,
                            weights={
                            description_space: sl.Param("desc_weight"),
                            energy_space: sl.Param("energy_weight")
                        }
                    )
                    .find(food_item)
                    .similar(description_space, sl.Param("query_text"))
                    .similar(energy_space, sl.Param("energy_intake_per_100g"))
                    .select_all()
                )
                result = app.query(query, query_text=desc_input, energy_intake_per_100g=energy_input, desc_weight=desc_weight, energy_weight=energy_weight)
                df_result = sl.PandasConverter.to_pandas(result)[["description", "food_category", "calories", 'similarity_score']]
                
                 # Calculate mean of the top 10 results
                mean_calories = df_result['calories'].head(10).mean() if not df_result.empty else 0
                st.write(f"🔍 Top results for: **{desc_input}**")
                top_10_results = df_result.head(10)
                st.write(f"📊 Mean calories of top 10 results: **{mean_calories:.2f}**")
                    # Display a bar chart for the mean calories
                               # Display a bar chart for the calories of the top 10 items
                st.bar_chart(top_10_results.set_index('description')['calories'])
                st.dataframe(top_10_results)
              
                
            except Exception as e:
                st.error(f"An error occurred: {e}")