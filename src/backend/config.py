from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── core ────────────────────────────────────────────────
    env: str = "dev"                               # dev / staging / prod
    data_path: Path = Path("data/sampled_food_db.parquet")
    umap_path: Path = Path("data/umap_df.parquet")
    embedding_model: str = "all-MiniLM-L6-v2"
    calories_min: int = 0
    calories_max: int = 1000

    # ─── meta ────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",           # auto‑load .env
        extra="ignore",            # ignore unknown keys
    )


settings = Settings()           