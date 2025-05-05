from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Dynamic .env Resolution ─────────────────────────────────────────────
ENV_NAME = os.getenv("ENV", "dev")  # fallback to 'dev'
DEFAULT_ENV_FILENAME = f".env.{ENV_NAME}"  # e.g. .env.dev, .env.prod

def get_env_file_path() -> str:
    """Get absolute path to the environment file relative to this config module."""
    return (Path(__file__).parent / DEFAULT_ENV_FILENAME).resolve().as_posix()


# ─── Settings Schema ─────────────────────────────────────────────────────
class Settings(BaseSettings):
    env: str = ENV_NAME
    data_path: Path = Path("data/sampled_food_db.parquet")
    umap_path: Path = Path("data/umap_df.parquet")
    embedding_model: str = "all-MiniLM-L6-v2"
    calories_min: int = 0
    calories_max: int = 1000

    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
