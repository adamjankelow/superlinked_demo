
"""
Generate and save UMAP vectors for food database embeddings.
"""

from backend.ingest.loader import load_data, build_superlinked_app
from backend.features.umap import create_umap_vectors
from backend.config import settings

def main():
    """Generate and save UMAP vectors for food database embeddings."""
    # 1) load data
    df = load_data()
    # 2) build superlinked app
    ctx = build_superlinked_app(df)
    # 3) create umap vectors
    umap_df = create_umap_vectors(ctx, df)
    # 4) save umap vectors
    umap_df.to_parquet(settings.umap_path)
    print("umap_df saved")

if __name__ == "__main__":
    main()
    
# python -m scripts.build_umap - run from the root directory