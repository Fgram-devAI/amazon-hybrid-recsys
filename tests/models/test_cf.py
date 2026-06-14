"""Tests for the Surprise-backed CF wrappers (tiny fixture)."""

import pandas as pd
import pytest

from src.models.cf import KNNRecommender, SVDRecommender

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i1", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i2", "rating": 5.0},
        {"user_id": "u3", "parent_asin": "i1", "rating": 3.0},
    ]
)


def test_svd_fits_and_predicts_in_range():
    model = SVDRecommender(random_state=42).fit(TRAIN)
    value = model.predict("u1", "i2")
    assert 1.0 <= value <= 5.0
    # unknown user/item still returns a clipped float (no crash)
    assert 1.0 <= model.predict("u_new", "i_new") <= 5.0


def test_knn_fits_and_predicts_in_range():
    model = KNNRecommender().fit(TRAIN)
    value = model.predict("u1", "i2")
    assert 1.0 <= value <= 5.0
    assert 1.0 <= model.predict("u_new", "i_new") <= 5.0


@pytest.fixture
def _toy_train_knn() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "B", "rating": 4.0, "timestamp": 2},
            {"user_id": "u2", "parent_asin": "A", "rating": 4.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "C", "rating": 5.0, "timestamp": 4},
            {"user_id": "u3", "parent_asin": "B", "rating": 5.0, "timestamp": 5},
            {"user_id": "u3", "parent_asin": "C", "rating": 4.0, "timestamp": 6},
        ]
    )


def test_knn_default_preserves_historical_msd_and_item_based() -> None:
    """No-args wrapper must keep the pre-change Surprise behavior (msd, item-based)."""
    model = KNNRecommender(k=2)
    sim_options = model._algo.sim_options  # type: ignore[attr-defined]
    # When no name is set, Surprise applies "msd" by default.
    assert sim_options.get("name", "msd") == "msd"
    assert sim_options["user_based"] is False


def test_knn_accepts_sim_name_cosine(_toy_train_knn) -> None:
    model = KNNRecommender(k=2, sim_name="cosine")
    sim_options = model._algo.sim_options  # type: ignore[attr-defined]
    assert sim_options["name"] == "cosine"
    assert sim_options["user_based"] is False
    model.fit(_toy_train_knn)
    rating = model.predict("u1", "C")
    assert 1.0 <= rating <= 5.0


def test_knn_accepts_sim_name_pearson_and_user_based_flag(_toy_train_knn) -> None:
    model = KNNRecommender(k=2, sim_name="pearson", user_based=True)
    sim_options = model._algo.sim_options  # type: ignore[attr-defined]
    assert sim_options["name"] == "pearson"
    assert sim_options["user_based"] is True


def test_knn_rejects_unknown_sim_name() -> None:
    with pytest.raises(ValueError):
        KNNRecommender(k=2, sim_name="not-a-similarity")
