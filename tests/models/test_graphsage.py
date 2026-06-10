"""GraphSAGE wrapper: forward/regression step, head output in [1,5], interface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.graphsage import GraphSAGERecommender


class _FakeEmbedder:
    name = "fake-embedder-v1"
    device = "cpu"
    def encode(self, texts):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 4)).astype(np.float32)


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":    ["u1", "u1", "u2", "u2", "u3"],
        "parent_asin":["i1", "i2", "i1", "i3", "i2"],
        "rating":     [5.0, 4.0, 5.0, 2.0, 5.0],
        "timestamp":  [1, 2, 3, 4, 5],
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


def test_graphsage_fits_and_predicts_in_rating_range():
    model = GraphSAGERecommender(
        hidden_dim=8, n_layers=2, epochs=2, lr=0.05,
        batch_size=4, seed=0, device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8, min_doc_freq=1,
    ).fit(_toy_train(), _toy_metadata())

    pred = model.predict("u1", "i3")
    assert 1.0 <= pred <= 5.0


def test_graphsage_recommend_returns_unseen_items_only():
    model = GraphSAGERecommender(
        hidden_dim=8, n_layers=2, epochs=2, lr=0.05,
        batch_size=4, seed=0, device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8, min_doc_freq=1,
    ).fit(_toy_train(), _toy_metadata())

    recs = model.recommend("u1", k=2, candidates=["i1", "i2", "i3"])
    assert recs == ["i3"]


def test_graphsage_predict_falls_back_for_unseen_user_and_item():
    model = GraphSAGERecommender(
        hidden_dim=8, n_layers=2, epochs=1, lr=0.05,
        batch_size=4, seed=0, device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8, min_doc_freq=1,
    ).fit(_toy_train(), _toy_metadata())

    pred = model.predict("ghost", "ghost")
    assert 1.0 <= pred <= 5.0


def test_graphsage_predict_uses_cached_embeddings():
    model = GraphSAGERecommender(
        hidden_dim=8, n_layers=2, epochs=1, lr=0.05,
        batch_size=4, seed=0, device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8, min_doc_freq=1,
    ).fit(_toy_train(), _toy_metadata())

    def _raise_if_reencoded(*args, **kwargs):
        raise AssertionError("predict() should reuse cached node embeddings")

    assert model._model is not None
    model._model.encode = _raise_if_reencoded

    pred = model.predict("u1", "i3")
    assert 1.0 <= pred <= 5.0


def test_graphsage_stores_weight_decay():
    model = GraphSAGERecommender(
        hidden_dim=8, n_layers=2, epochs=1, lr=0.05,
        weight_decay=1e-5, batch_size=4, seed=0, device="cpu",
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8, min_doc_freq=1,
    )
    assert model.weight_decay == 1e-5
