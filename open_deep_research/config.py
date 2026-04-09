from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    model: str = "ollama/llama3.1"
    api_base: str | None = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.3


class SearXNGConfig(BaseModel):
    base_url: str = "http://localhost:8888"


class SearchConfig(BaseModel):
    provider: str = "searxng"
    searxng: SearXNGConfig = SearXNGConfig()


class ResearchConfig(BaseModel):
    max_iterations: int = 10
    max_sources: int = 30
    budget_tokens: int = 500_000
    source_summary_tokens: int = 500
    follow_links: bool = True
    max_followed_links: int = 5


class OutputConfig(BaseModel):
    format: str = "markdown"
    include_confidence: bool = True
    include_contradictions: bool = True
    verbose: bool = False


class SessionsConfig(BaseModel):
    storage_dir: str = "~/.open-deep-research/sessions"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ODR_", env_nested_delimiter="__")

    llm: LLMConfig = LLMConfig()
    search: SearchConfig = SearchConfig()
    research: ResearchConfig = ResearchConfig()
    output: OutputConfig = OutputConfig()
    sessions: SessionsConfig = SessionsConfig()


def load_settings(config_path: Path | None = None) -> Settings:
    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    # Try default location
    default = Path("config.yaml")
    if default.exists():
        with open(default) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
