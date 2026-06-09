"""Weighted hybrid with optional per-component z-score calibration.

SVD and content scores live on different distributions; without calibration the
component with larger natural spread dominates the blend. With ``calibrate=True``
the hybrid standardizes each component's score against its training-time mean
and std, then maps the standardized score onto the global training-rating mean
and standard deviation before blending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import Recommender


class CalibratedHybrid(Recommender):
    def __init__(
        self,
        cf: Recommender,
        content: Recommender,
        *,
        alpha: float = 0.5,
        calibrate: bool = True,
        calibration_max_rows: int | None = None,
        random_state: int = 42,
        progress: bool = False,
    ) -> None:
        super().__init__()
        self.cf = cf
        self.content = content
        self.alpha = float(alpha)
        self.calibrate = bool(calibrate)
        self.calibration_max_rows = calibration_max_rows
        self.random_state = int(random_state)
        self.progress = bool(progress)
        self.cf_mean_: float = 0.0
        self.cf_std_: float = 1.0
        self.content_mean_: float = 0.0
        self.content_std_: float = 1.0
        self.target_std_: float = 1.0

    def fit(
        self, train: pd.DataFrame, metadata: object = None
    ) -> "CalibratedHybrid":
        self._fit_means(train)
        if not self.cf.is_fitted:
            self.cf.fit(train, metadata)
        if not self.content.is_fitted:
            self.content.fit(train, metadata)

        if self.calibrate:
            calibration_rows = train
            if (
                self.calibration_max_rows is not None
                and len(calibration_rows) > self.calibration_max_rows
            ):
                calibration_rows = calibration_rows.sample(
                    n=self.calibration_max_rows,
                    random_state=self.random_state,
                )
            cf_pairs = zip(calibration_rows["user_id"], calibration_rows["parent_asin"])
            cf_scores = np.array(
                [
                    self.cf.predict(u, i)
                    for u, i in tqdm(
                        cf_pairs,
                        total=len(calibration_rows),
                        desc="[calibrated_hybrid] cf calibration",
                        unit="row",
                        disable=not self.progress,
                    )
                ],
                dtype=float,
            )
            content_pairs = zip(calibration_rows["user_id"], calibration_rows["parent_asin"])
            content_scores = np.array(
                [
                    self.content.predict(u, i)
                    for u, i in tqdm(
                        content_pairs,
                        total=len(calibration_rows),
                        desc="[calibrated_hybrid] content calibration",
                        unit="row",
                        disable=not self.progress,
                    )
                ],
                dtype=float,
            )
            self.cf_mean_ = float(cf_scores.mean())
            self.content_mean_ = float(content_scores.mean())
            self.cf_std_ = float(cf_scores.std() or 1.0)
            self.content_std_ = float(content_scores.std() or 1.0)
            self.target_std_ = float(calibration_rows["rating"].std(ddof=0) or 1.0)
        return self

    def _calibrate(self, raw: float, mean: float, std: float) -> float:
        z = (raw - mean) / (std or 1.0)
        return self.global_mean_ + z * self.target_std_

    def predict(self, user_id: str, parent_asin: str) -> float:
        cf_raw = self.cf.predict(user_id, parent_asin)
        content_raw = self.content.predict(user_id, parent_asin)
        if self.calibrate:
            cf_val = self._calibrate(cf_raw, self.cf_mean_, self.cf_std_)
            content_val = self._calibrate(content_raw, self.content_mean_, self.content_std_)
        else:
            cf_val = cf_raw
            content_val = content_raw
        return self._clip(self.alpha * cf_val + (1.0 - self.alpha) * content_val)
