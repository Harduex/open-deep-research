from pathlib import Path

from open_deep_research.config import Settings, load_settings


def test_default_settings():
    s = Settings()
    assert s.llm.model == "ollama/llama3.1"
    assert s.search.provider == "searxng"
    assert s.research.max_iterations == 10
    assert s.research.max_sources == 30
    assert s.research.budget_tokens == 500_000


def test_load_settings_no_config():
    s = load_settings(None)
    assert isinstance(s, Settings)


def test_load_settings_missing_file():
    s = load_settings(Path("/nonexistent/config.yaml"))
    assert isinstance(s, Settings)
