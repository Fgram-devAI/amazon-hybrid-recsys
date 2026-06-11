"""GraphRecommender base: device pick, checkpoint round-trip, fallback predict."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
import torch

from src.models.graph_base import GraphRecommender, pick_device


def test_pick_device_returns_cpu_when_no_accelerator_requested():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_falls_back_to_cpu_when_cuda_and_mps_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto") == torch.device("cpu")


class _DummyGraph(GraphRecommender):
    """Concrete subclass that only implements the trivial bits for interface checks."""

    def __init__(self):
        super().__init__()
        self._weights = torch.tensor([1.23])

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "_DummyGraph":
        self._fit_means(train)
        return self

    def predict(self, user_id: str, parent_asin: str) -> float:
        return self._clip(self._fallback(user_id, parent_asin))

    def state_dict(self) -> dict[str, object]:
        return {"weights": self._weights.clone()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._weights = cast(torch.Tensor, state["weights"])


def test_predict_falls_back_for_unseen_user_and_item():
    train = pd.DataFrame({
        "user_id": ["u1"], "parent_asin": ["i1"], "rating": [4.0], "timestamp": [1],
    })
    model = _DummyGraph().fit(train)
    pred = model.predict("ghost_user", "ghost_item")
    assert 1.0 <= pred <= 5.0


def test_checkpoint_save_load_round_trips(tmp_path):
    train = pd.DataFrame({
        "user_id": ["u1"], "parent_asin": ["i1"], "rating": [4.0], "timestamp": [1],
    })
    model = _DummyGraph().fit(train)
    path = tmp_path / "checkpoint.pt"
    model.save_checkpoint(path)

    restored = _DummyGraph().fit(train)
    restored._weights = torch.tensor([0.0])   # corrupt before load
    restored.load_checkpoint(path)
    assert restored._weights.item() == pytest.approx(1.23)
