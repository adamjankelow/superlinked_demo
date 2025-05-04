# backend/umap_cache.py
"""
Creates / loads a UMAP projection and stores it as a Parquet next to the raw
embeddings file.  Idempotent: runs UMAP once, then reloads.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from superlinked import framework as sl
from ..config import settings  # <-- uses settings.data_path & settings.cache_dir


def get_umap_df() -> pd.DataFrame:
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
    
    #add the metadata to the umap_df
    umap_df = umap_df.join(df.set_index('fdc_id')[['description', 'food_category', 'calories']], how='inner')

    return umap_df


