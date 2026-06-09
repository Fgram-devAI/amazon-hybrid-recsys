"""Tests for the calibrated hybrid (alpha + per-component z-score)."""

import pandas as pd
import pytest

from src.models.base import Recommender
from src.models.calibrated_hybrid import CalibratedHybrid

_TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 3.0},
    ]
)


class ConstantStub(Recommender):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return self.value


class CountingStub(ConstantStub):
    def __init__(self, value: float) -> None:
        super().__init__(value)
        self.calls = 0

    def predict(self, user_id, parent_asin):
        self.calls += 1
        return self.value


class ItemScoreStub(Recommender):
    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__()
        self.scores = scores

    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return self.scores[parent_asin]


def test_calibration_disabled_blends_exactly_like_weighted_hybrid():
    hybrid = CalibratedHybrid(
        ConstantStub(4.0), ConstantStub(2.0), alpha=0.5, calibrate=False
    ).fit(_TRAIN)
    assert hybrid.predict("u1", "i1") == pytest.approx(3.0)


def test_calibration_uses_training_mean_when_components_differ():
    # When calibrate=True the blend uses z-scores recentred to the global rating mean.
    cf = ConstantStub(4.0)
    content = ConstantStub(2.0)
    hybrid = CalibratedHybrid(cf, content, alpha=0.5, calibrate=True).fit(_TRAIN)
    # Constant predictions have zero training spread -> z-score is 0, so the
    # calibrated output equals the global training mean (4 in this fixture).
    assert hybrid.predict("u1", "i1") == pytest.approx(hybrid.global_mean_)


def test_calibration_rescales_to_training_rating_spread():
    train = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
            {"user_id": "u1", "parent_asin": "i2", "rating": 1.0},
        ]
    )
    component = ItemScoreStub({"i1": 2.0, "i2": 4.0})
    hybrid = CalibratedHybrid(
        component,
        ConstantStub(3.0),
        alpha=1.0,
        calibrate=True,
    ).fit(train)

    assert hybrid.target_std_ == pytest.approx(2.0)
    assert hybrid.predict("u1", "i2") == pytest.approx(5.0)


def test_alpha_endpoints_pick_the_right_component_uncalibrated():
    cf = ConstantStub(4.5)
    content = ConstantStub(1.5)
    assert (
        CalibratedHybrid(cf, content, alpha=1.0, calibrate=False)
        .fit(_TRAIN)
        .predict("u1", "i1")
        == 4.5
    )
    assert (
        CalibratedHybrid(cf, content, alpha=0.0, calibrate=False)
        .fit(_TRAIN)
        .predict("u1", "i1")
        == 1.5
    )


def test_calibration_can_sample_training_rows_to_bound_fit_cost():
    train = pd.DataFrame(
        [
            {"user_id": f"u{i}", "parent_asin": f"i{i}", "rating": 4.0}
            for i in range(20)
        ]
    )
    cf = CountingStub(4.0)
    content = CountingStub(3.0)
    CalibratedHybrid(
        cf,
        content,
        alpha=0.5,
        calibrate=True,
        calibration_max_rows=5,
        random_state=42,
    ).fit(train)
    assert cf.calls == 5
    assert content.calls == 5
