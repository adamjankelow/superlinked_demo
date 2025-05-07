from pathlib import Path
import os
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Dynamic .env Resolution ─────────────────────────────────────────────
ENV_NAME           = os.getenv("ENV", "dev")
DEFAULT_ENV_FILENAME = f".env.{ENV_NAME}"

def get_env_file_path() -> str:
    return (Path(__file__).parent / DEFAULT_ENV_FILENAME).resolve().as_posix()


class Settings(BaseSettings):
    # environment overrides 
    env:              str  = Field(default=ENV_NAME)
    data_path:        Path = Field(default=Path("data/sampled_food_db.parquet"))
    umap_path:        Path = Field(default=Path("data/umap_df.parquet"))
    embedding_model:  str  = Field(default="all-MiniLM-L6-v2")

    # ─── app defaults ────────────────────────────────────────────────────
    calories_min:           int = Field(default=0)
    calories_max:           int = Field(default=1000)
    umap_top_n_food_items:  int = Field(default=10)

    # ─── validation ─────────────────────────────────────────────────────
    @field_validator("data_path", "umap_path")
    def _validate_paths_exist(cls, v: Path) -> Path:
        """
        Fail fast if the configured file paths do not actually exist on disk.
        This catches typos in your .env early, rather than 5 flavors deep in code.
        """
        resolved = v.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Configured path not found: {resolved}")
        return resolved

    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()