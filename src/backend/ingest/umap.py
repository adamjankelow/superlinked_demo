
"""
Holds functions for creating and retrieving UMAP vectors
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from pathlib import Path
import pandas as pd
import umap
from superlinked import framework as sl
from ..config import settings  


def load_umap_df() -> pd.DataFrame:
    """
    Returns a DataFrame with `umap_x`, `umap_y` + the original metadata.
    """

    repo_root = Path(__file__).resolve().parents[3]
    umap_file = repo_root / settings.umap_path
    df = pd.read_parquet(umap_file)
    return df

def create_umap_vectors(app, index, food_item, df:pd.DataFrame):
    
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
 
    vs = sl.VectorSampler(app=app)
    vector_collection = vs.get_all_vectors(index, food_item)
    vectors = vector_collection.vectors
    vector_df = pd.DataFrame(vectors, index=[int(id_) for id_ in vector_collection.id_list])
    
    # Transform vectors using UMAP
    umap_transform = umap.UMAP(random_state=0, transform_seed=0, n_jobs=1, metric="cosine")
    umap_transform = umap_transform.fit(vector_df)
    umap_vectors = umap_transform.transform(vector_df)
    umap_df = pd.DataFrame(umap_vectors, columns=["dimension_1", "dimension_2"], index=vector_df.index)
    
    #add the metadata to the umap_df
    umap_df = umap_df.join(df.set_index('fdc_id')[['description', 'food_category', 'calories']], how='inner')

    return umap_df



def plot_umap_scatter(umap_df: pd.DataFrame) -> plt.Figure:
    """
    Plots a UMAP scatter plot for the top 10 food items based on search results.

    Args:
        umap_df (pd.DataFrame): DataFrame containing UMAP coordinates and food metadata.
            Expected columns: ['dimension_1', 'dimension_2', 'food_category', 'description']

    Returns:
        plt.Figure: The matplotlib figure object containing the scatter plot.
    """
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
    
    return plt


    # Display the plot in the Streamlit app
  



