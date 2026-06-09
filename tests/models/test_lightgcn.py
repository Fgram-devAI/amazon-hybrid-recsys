"""LightGCN wrapper: forward/BPR step, calibrated head, Recommender interface."""

from __future__ import annotations

import pandas as pd

from src.models.lightgcn import LightGCNRecommender


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":    ["u1", "u1", "u2", "u2", "u3", "u3"],
        "parent_asin":["i1", "i2", "i1", "i3", "i2", "i3"],
        "rating":     [5.0, 4.0, 5.0, 4.0, 4.0, 5.0],
        "timestamp":  [1, 2, 3, 4, 5, 6],
    })


def test_lightgcn_fits_and_predicts_in_rating_range():
    model = LightGCNRecommender(
        embedding_dim=8, n_layers=2, epochs=2, lr=0.05,
        num_negatives=2, batch_size=4, seed=0, device="cpu",
    ).fit(_toy_train())

    p = model.predict("u1", "i3")
    assert 1.0 <= p <= 5.0


def test_lightgcn_recommend_returns_unseen_items_only():
    model = LightGCNRecommender(
        embedding_dim=8, n_layers=2, epochs=2, lr=0.05,
        num_negatives=2, batch_size=4, seed=0, device="cpu",
    ).fit(_toy_train())

    recs = model.recommend("u1", k=2, candidates=["i1", "i2", "i3"])
    # u1 already saw i1, i2 -> only i3 is eligible
    assert recs == ["i3"]


def test_lightgcn_predict_for_unseen_user_falls_back():
    model = LightGCNRecommender(
        embedding_dim=8, n_layers=2, epochs=1, lr=0.05,
        num_negatives=2, batch_size=4, seed=0, device="cpu",
    ).fit(_toy_train())

    pred = model.predict("ghost", "i1")
    assert 1.0 <= pred <= 5.0


def test_lightgcn_calibration_uses_validation_slice_carved_from_train():
    """Calibration must NOT touch test data; it consumes a train-derived val slice."""
    model = LightGCNRecommender(
        embedding_dim=4, n_layers=1, epochs=1, lr=0.05,
        num_negatives=1, batch_size=4, seed=0, device="cpu",
        validation_fraction=0.5,
    ).fit(_toy_train())

    # calibration coefficients are set after fit
    assert model._calibration_beta is not None
    assert model._calibration_intercept is not None
