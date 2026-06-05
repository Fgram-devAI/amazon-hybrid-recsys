"""Load and query the pipeline configuration (config/config.yaml)."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """Parse the YAML config file into a dict."""
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def active_dataset_category(config: dict[str, Any]) -> str:
    """Return the Amazon category name for the currently active dataset."""
    active = config["active_dataset"]
    return config["datasets"][active]["category"]
