
# 2) import & call
import pandas as pd
import superlinked as sl
from src.backend.utils.data import load_data, build_superlinked_app
from src.backend.utils.umap import create_umap_vectors

def main():
    # 1) load data
    df = load_data()
    # 2) build superlinked app
    app, index, food_item, description_space, food_category_text_space, food_category_categorical_space, calories_space = build_superlinked_app(df)
    
    # 3) create umap vectors
    umap_df = create_umap_vectors(app, index, food_item, df)
    print("umap_df created")
    # 4) save umap vectors
    umap_df.to_parquet('data/umap_df.parquet')
    print("umap_df saved")

if __name__ == "__main__":
    main()
    
# python -m scripts.build_umap - run from the root directory