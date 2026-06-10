"""GraphSAGE-BPR wrapper: ranking objective over enriched graph node features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.graphsage_bpr import GraphSAGEBPRRecommender


class _FakeEmbedder:
    name = "fake-embedder-v1"
    device = "cpu"

    def encode(self, texts):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 4)).astype(np.float32)


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "parent_asin": ["i1", "i2", "i1", "i3", "i2"],
        "rating": [5.0, 4.0, 5.0, 2.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5],
    })


def _toy_metadata() -> pd.DataFrame:
    return pd.DataFrame({
        "parent_asin": ["i1", "i2", "i3"],
        "title": ["a", "b", "c"],
        "description": ["d1", "d2", "d3"],
        "categories": [["Action"], ["Comedy"], ["Action"]],
        "price": [10.0, 20.0, 30.0],
        "average_rating": [4.0, 4.5, 3.5],
        "rating_number": [100, 50, 75],
    })


def _model(**kwargs) -> GraphSAGEBPRRecommender:
    return GraphSAGEBPRRecommender(
        hidden_dim=8,
        n_layers=2,
        epochs=1,
        lr=0.05,
        num_negatives=2,
        batch_size=4,
        seed=0,
        device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        **kwargs,
    )


def test_graphsage_bpr_fits_and_predicts_in_rating_range():
    model = _model().fit(_toy_train(), _toy_metadata())

    pred = model.predict("u1", "i3")
    assert 1.0 <= pred <= 5.0


def test_graphsage_bpr_recommend_returns_unseen_items_only():
    model = _model().fit(_toy_train(), _toy_metadata())

    recs = model.recommend("u1", k=2, candidates=["i1", "i2", "i3"])
    assert recs == ["i3"]


def test_graphsage_bpr_checkpoint_state_keeps_calibration_and_embeddings():
    model = _model().fit(_toy_train(), _toy_metadata())
    state = model.state_dict()

    assert "calibration_beta" in state
    assert "calibration_intercept" in state
    assert state["final_embeddings"] is not None
