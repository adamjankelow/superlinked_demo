import pandas as pd
import os
import streamlit as st
from superlinked import framework as sl
import numpy as np
from utills import create_umap_df, plot_umap_scatter

cols_to_display = ["description", "food_category", "calories", 'similarity_score']
def simple_search(food_item, description_space: str, index: object, app: object) -> None:
    """
    Perform a simple search for food items based on their descriptions.

    Parameters:
    - food_item (str): The food item schema.
    - description_space (str): The description space for similarity.
    - index (object): The index used for querying.
    - app (object): The application instance for querying.

    Returns:
    - None: Displays results directly in the Streamlit app.
    """
    st.markdown("Search for food items. Just type a query below to explore matches based on the item's description.")

    query_input = st.text_input("Food description", "cereal with sugar")
    
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
                df_result = sl.PandasConverter.to_pandas(result)[cols_to_display]
                st.write(f"🔍 Top results for: **{query_input}**")
                st.dataframe(df_result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

def weighted_search(food_item , description_space: str, food_category_text_space: str, food_category_categorical_space: str, index: object, app: object) -> None:
    """
    Perform a weighted search for food items based on user-defined weights.

    Parameters:
    - food_item (str): The food item schema.
    - description_space (str): The description space for similarity.
    - food_category_text_space (str): The text space for food categories.
    - food_category_categorical_space (str): The categorical space for food categories.
    - index (object): The index used for querying.
    - app (object): The application instance for querying.

    Returns:
    - None: Displays results directly in the Streamlit app.
    """
    st.markdown("Superlinked supports weighted search. Use the sliders below to adjust the weights of the description and category search spaces to see how they affect the results.")
    query = st.text_input("Food description", "apple")
    food_category = st.text_input("Food category", "dessert")
    desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
    cat_weight = st.slider("Category weight", -3.0, 3.0, 1.0)

    if query:
        q = (
            sl.Query(index, weights={
                description_space: sl.Param("desc_weight"),
                food_category_text_space: sl.Param("cat_weight"),
                food_category_categorical_space: sl.Param("cat_weight")
            })
            .find(food_item)
            .similar(description_space, sl.Param("food_item"))
            .similar(food_category_text_space, sl.Param("food_category"))
            .similar(food_category_categorical_space, sl.Param("query_text"))
            .select_all()
        )
        result = app.query(q, food_item=query, food_category=food_category, desc_weight=desc_weight, cat_weight=cat_weight)
        
        result_df = sl.PandasConverter.to_pandas(result)
    
        result_df.sort_values(by='similarity_score', ascending=False, inplace=True)
        top_10_results = result_df.head(10)
         
        st.dataframe(top_10_results)
        umap_df = create_umap_df(app, index, food_item, top_10_results)
        plot_umap_scatter(umap_df)
       
        return result_df

def numeric_search(food_item , description_space: str, calories_space: str, index: object, app: object) -> None:
    """
    Perform a numeric search for food items based on calorie values.

    Parameters:
    - food_item (str): The food item schema.
    - description_space (str): The description space for similarity.
    - calories_space (str): The space for calorie values.
    - index (object): The index used for querying.
    - app (object): The application instance for querying.

    Returns:
    - None: Displays results directly in the Streamlit app.
    """
    st.markdown("We can also include numeric spaces in our search. See how the results change as we change the weights of the description and calories. Include a bar to show the calories of the top 10 results.")
    desc_input = st.text_input("Food description", "chicken")
    calories_input = st.number_input("Calories per 100g", min_value=0, max_value=1000, value=200)
    desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
    calories_weight = st.slider("Calories weight", -3.0, 3.0, 1.0)

    if desc_input and calories_input:
        with st.spinner('Searching...'):
            try:
                query = (
                    sl.Query(index, weights={
                        description_space: sl.Param("desc_weight"),
                        calories_space: sl.Param("calories_weight")
                    })
                    .find(food_item)
                    .similar(description_space, sl.Param("query_text"))
                    .similar(calories_space, sl.Param("calories_intake_per_100g"))
                    .select_all()
                )
                result = app.query(query, query_text=desc_input, calories_intake_per_100g=calories_input, desc_weight=desc_weight, calories_weight=calories_weight)
                df_result = sl.PandasConverter.to_pandas(result)[cols_to_display]
                
                # Calculate mean of the top 10 results
                mean_calories = df_result['calories'].head(10).mean() if not df_result.empty else 0
                st.write(f"🔍 Top results for: **{desc_input}**")
                top_10_results = df_result.head(10)
                st.write(f"📊 Mean calories of top 10 results: **{mean_calories:.2f}**")
                
                # Display a bar chart for the calories of the top 10 items
                st.bar_chart(top_10_results.set_index('description')['calories'])
                st.dataframe(top_10_results)
              
            except Exception as e:
                st.error(f"An error occurred: {e}")

def combined_search(food_item , description_space: str, food_category_categorical_space: str, calories_space: str, index: object, app: object, categories: list) -> None:
    """
    Combine categorical, numerical, and text search for food items.

    Parameters:
    - food_item (str): The food item schema.
    - description_space (str): The description space for similarity.
    - food_category_categorical_space (str): The categorical space for food categories.
    - calories_space (str): The energy space for calorie values.
    - index (object): The index used for querying.
    - app (object): The application instance for querying.
    - categories (list): List of food categories for selection.

    Returns:
    - None: Displays results directly in the Streamlit app.
    """
    st.markdown("Combining categorical, numerical and text search. We use a hard filtering for the categorical space.")
    
    food_category_input = st.selectbox("Food category", categories)
    desc_input = st.text_input("Food description")
    calories_input = st.number_input("Calories per 100g", min_value=0, max_value=1000)
    
    query = (
        sl.Query(index, weights={
            description_space: sl.Param("desc_weight"),
            calories_space: sl.Param("calories_weight")
        })
        .find(food_item)
        .similar(food_category_categorical_space.category, sl.Param("query_categories"))
        .similar(description_space, sl.Param("query_text"))
        .similar(calories_space, sl.Param("calories_intake_per_100g"))
        .select_all()
    )
    
    result = app.query(query, query_categories=food_category_input, query_text=desc_input, calories_intake_per_100g=calories_input, desc_weight=1.5, calories_weight=1)
    
    result_df = sl.PandasConverter.to_pandas(result)[cols_to_display]
    
    if len(result_df) > 0:
        st.dataframe(result_df)
    else:
        st.error("No results found. Try changing the search parameters.")
        
    return result_df