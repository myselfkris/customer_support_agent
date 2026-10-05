"""Central application configuration.

Reads settings from environment variables and `.env` (repo root).
Keeps secrets and tunables out of source code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (one level above app/)
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini
    gemini_api_key: str = ""
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"

    # PostgreSQL / pgvector (used by the RAG pipeline)
    database_url: str = "postgresql://postgres:password@localhost:5432/customer_support"

    # Observability / logging
    log_level: str = "info"

    # Auth (Chunk 5): API keys for customers, JWT for admins
    jwt_secret: str = ""           # set a long random value in .env (JWT_SECRET)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    api_key_prefix: str = "cust_live_"

    @property
    def llm_configured(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
