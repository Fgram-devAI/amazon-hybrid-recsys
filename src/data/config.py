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


def dataset_k_core(config: dict[str, Any]) -> int:
    """k-core threshold for the active dataset, with per-dataset override.

    Sparse categories (e.g. Digital_Music) need a lower core than the global
    default, so a dataset may set its own ``k_core``; otherwise the global
    ``preprocessing.k_core`` applies.
    """
    active = config["active_dataset"]
    dataset = config["datasets"][active]
    return dataset.get("k_core", config["preprocessing"]["k_core"])
