"""Tests for loading the pipeline configuration."""

from src.data.config import load_config, active_dataset_category


def test_loads_config_and_resolves_active_category():
    cfg = load_config("config/config.yaml")
    assert cfg["active_dataset"] == "video_games"
    assert cfg["amazon_base_url"].startswith("https://")
    assert active_dataset_category(cfg) == "Video_Games"
