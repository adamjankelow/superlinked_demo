
"""
Generate and save UMAP vectors for food database embeddings.
"""

from src.backend.utils.data import load_data, build_superlinked_app
from src.backend.utils.umap import create_umap_vectors
from src.backend.config import settings

def main():
    """Generate and save UMAP vectors for food database embeddings."""
    # 1) load data
    df = load_data()
    # 2) build superlinked app
    app, index, food_item, description_space, food_category_text_space, food_category_categorical_space, calories_space = build_superlinked_app(df)
    
    # 3) create umap vectors
    umap_df = create_umap_vectors(app, index, food_item, df)
    print("umap_df created")
    # 4) save umap vectors
    umap_df.to_parquet(settings.umap_path)
    print("umap_df saved")

if __name__ == "__main__":
    main()
    
# python -m scripts.build_umap - run from the root directory