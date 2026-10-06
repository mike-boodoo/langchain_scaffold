from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.5", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    persist_dir: Path = Field(
        default=Path("./storage/index"),
        alias="PERSIST_DIR",
    )
    mcp_url: str = Field(
        default="http://127.0.0.1:8000/mcp",
        alias="MCP_URL",
    )

    max_context_chars: int = Field(
        default=18_000,
        alias="MAX_CONTEXT_CHARS",
        ge=1_000,
        le=100_000,
    )
    max_tool_calls: int = Field(
        default=20,
        alias="MAX_TOOL_CALLS",
        ge=1,
        le=100,
    )

    @field_validator("openai_model", "openai_embedding_model", mode="before")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("model name cannot be empty")
        return value

    @model_validator(mode="after")
    def initialize_paths(self) -> "Settings":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
