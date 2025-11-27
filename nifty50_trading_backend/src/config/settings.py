import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings pulled from environment variables with defaults."""
    ENV: str = Field(default="development", description="Runtime environment")
    DATA_DIR: str = Field(default="storage/data", description="Directory to store raw data")
    FEATURES_DIR: str = Field(default="storage/features", description="Directory to store engineered features")
    MODEL_DIR: str = Field(default="storage/models", description="Directory to store supervised models")
    RL_DIR: str = Field(default="storage/rl", description="Directory to store RL policies and sims")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    CORS_ALLOW_ORIGINS: list[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")

    # Optional broker keys
    BROKER_API_KEY: str | None = Field(default=None, description="Broker API key")
    BROKER_API_SECRET: str | None = Field(default=None, description="Broker API secret")

    # Pydantic v2 settings configuration (replaces class Config in v1)
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        # Ignore unrelated environment variables injected by the platform to prevent ValidationError
        extra="ignore",
    )


@lru_cache
# PUBLIC_INTERFACE
def get_settings() -> Settings:
    """Return cached settings instance and ensure storage directories exist."""
    s = Settings()
    for d in [s.DATA_DIR, s.FEATURES_DIR, s.MODEL_DIR, s.RL_DIR]:
        os.makedirs(d, exist_ok=True)
    return s
