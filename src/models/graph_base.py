"""Shared base for graph recommenders (LightGCN, GraphSAGE)."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch

from src.models.base import Recommender


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve device per spec: cuda -> mps -> cpu. ``requested`` may force a specific one."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GraphRecommender(Recommender):
    """Adds checkpoint save/load and a shared device attribute on top of Recommender."""

    def __init__(self, device: str = "auto") -> None:
        super().__init__()
        self.device = pick_device(device)

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.load_state_dict(state)
