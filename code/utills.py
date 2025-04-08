import pandas as pd
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from adjustText import adjust_text
from superlinked import framework as sl
from huggingface_hub import snapshot_download
# ---- Load Data ----
@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    
    # Construct the absolute path to the dataset
    dataset_path = os.path.join(current_dir, "../data/sampled_food_db.parquet")
    absolute_path = os.path.abspath(dataset_path)
    df = pd.read_parquet(absolute_path)  # replace with your dataset path
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
    
    # model_path = Path("PhenoDS/superlinked_demo/models/all-MiniLM-L6-v2").resolve()

    # description_space = sl.TextSimilaritySpace(
    #     text=food_item.description,
    #     model="all-MiniLM-L6-v2",
    #     model_cache_dir=model_path)  # must be a Path, not str
        
    # food_category_text_space = sl.TextSimilaritySpace(
    #     text= food_item.food_category,
    #     model="all-MiniLM-L6-v2",
    #     model_cache_dir=model_path)  # must be a Path, not str
    
  
    model_path = Path(snapshot_download("sentence-transformers/all-MiniLM-L6-v2"))

    # Create the similarity space using the schema field
    description_space = sl.TextSimilaritySpace(
            text=food_item.description,  # <-- must be a valid schema field
            model="all-MiniLM-L6-v2",
            model_cache_dir=model_path
        )
    
   
    food_category_text_space = sl.TextSimilaritySpace(text=food_item.food_category, model="all-MiniLM-L6-v2", model_cache_dir=model_path)

    # # Spaces
    # description_space = sl.TextSimilaritySpace(text=food_item.description, model="../models/all-MiniLM-L6-v2")
    # # Semantic similarity over food category text
    # food_category_text_space = sl.TextSimilaritySpace(text=food_item.food_category, model="../models/all-MiniLM-L6-v2")

    # Exact/category-level similarity (discrete match)
    food_category_categorical_space = sl.CategoricalSimilaritySpace(
        category_input=food_item.food_category,
        categories=categories
    )
    calories_space = sl.NumberSpace(
        food_item.calories,
        min_value=0,
        max_value=1000,
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


def create_umap_df(app, index, food_item , results_df:pd.DataFrame, n_results:int=20):
    """
    Create a DataFrame with UMAP transformed vectors and food metadata
    
    Args:
        app: Superlinked app instance
        index: Index name
        food_item: Food item name
        results_df: DataFrame containing search results
        n_results: Number of top results to include
        
    Returns:
        DataFrame containing UMAP coordinates and food metadata
    """
    # Collect all vectors from the app
    import umap
    vs = sl.VectorSampler(app=app)
    vector_collection = vs.get_all_vectors(index, food_item)
    vectors = vector_collection.vectors
    vector_df = pd.DataFrame(vectors, index=[int(id_) for id_ in vector_collection.id_list])
    
    # Transform vectors using UMAP
    umap_transform = umap.UMAP(random_state=0, transform_seed=0, n_jobs=1, metric="cosine")
    umap_transform = umap_transform.fit(vector_df)
    umap_vectors = umap_transform.transform(vector_df)
    umap_df = pd.DataFrame(umap_vectors, columns=["dimension_1", "dimension_2"], index=vector_df.index)

    # Join with results metadata
    results_df.id = results_df.id.astype(int)
    umap_df = umap_df.join(results_df.head(n_results).set_index('id')[['description', 'food_category']], how='inner')
    
    return umap_df

# Create UMAP DataFrame

def plot_umap_scatter(umap_df):
    # Add a title and introductory text
    st.title("UMAP Visualization of Food Items")
    st.write("Here you can see the UMAP transformed vectors of the top 10 food items from the search results. Each point represents a food item, colored by its category.")

    # Create the figure
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='dimension_1', 
        y='dimension_2', 
        hue='food_category', 
        data=umap_df.head(10), 
        s=100,  # Size of the dots
        palette='viridis'
    )

    # Collect text objects for adjustment
    texts = []
    for i, row in umap_df.head(10).iterrows():
        text = plt.text(row['dimension_1'] + 0.1, row['dimension_2'], row['description'], fontsize=9)
        texts.append(text)

    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.title('UMAP Transformed Vectors of top 10 food items of search results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Food Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(plt)
