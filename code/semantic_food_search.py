import streamlit as st
import pandas as pd
from superlinked import framework as sl
import sys
from utills import load_data, build_superlinked_app

# ---- Streamlit UI ----
st.title("🥦 Semantic Search on Food Database")


# Sidebar for additional options
# st.sidebar.header("Search Options")


# Load data and build app
df = load_data()
app, index, food_item, description_space, food_category_text_space, food_category_categorical_space = build_superlinked_app(df)

mode = st.sidebar.radio(
    "Select Search Mode",
    ["Simple Search", "Weighted Search", "Category Filter", "Recommendations"]
)

if mode == "Simple Search":
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
                
if mode == "Weighted Search":
    st.markdown("Superlinked supports weighted search. Use the sliders below to adjust the weights of the description and category search spaces to see how they affect the results.")
    query = st.text_input("Search for a food")
    food_category = st.text_input("Search for a food category")
    desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
    cat_weight = st.slider("Category weight", -3.0, 3.0, 1.0)

    if query:
        q = (
            sl.Query(index, weights={description_space: sl.Param("w1"), food_category_text_space: sl.Param("w2"), food_category_categorical_space: sl.Param("w3")})
            .find(food_item)
            .similar(description_space, sl.Param("food_item"))
            .similar(food_category_text_space, sl.Param("food_category"))
            .similar(food_category_categorical_space, sl.Param("query_text"))
            .select_all()
        )
        result = app.query(q, food_item=query, food_category=food_category, w1=desc_weight, w2=cat_weight)
        st.dataframe(sl.PandasConverter.to_pandas(result))

                
