"""Tests for loading the pipeline configuration."""

from src.data.config import load_config, active_dataset_category, dataset_k_core


def test_loads_config_and_resolves_active_category():
    cfg = load_config("config/config.yaml")
    assert cfg["active_dataset"] == "video_games"
    assert cfg["amazon_base_url"].startswith("https://")
    assert active_dataset_category(cfg) == "Video_Games"


def test_dataset_k_core_prefers_per_dataset_override_then_global():
    cfg = {
        "active_dataset": "digital_music",
        "preprocessing": {"k_core": 5},
        "datasets": {
            "digital_music": {"category": "Digital_Music", "k_core": 2},
            "video_games": {"category": "Video_Games"},  # no override
        },
    }
    # very sparse dataset overrides the global core
    assert dataset_k_core(cfg) == 2
    # dense dataset with no override falls back to the global core
    cfg["active_dataset"] = "video_games"
    assert dataset_k_core(cfg) == 5
