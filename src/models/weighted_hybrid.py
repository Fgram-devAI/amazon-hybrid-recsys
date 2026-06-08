"""Weighted hybrid: alpha * CF + (1 - alpha) * content."""

from .base import Recommender


class WeightedHybrid(Recommender):
    def __init__(self, cf, content, alpha=0.5):
        super().__init__()
        self.cf = cf
        self.content = content
        self.alpha = float(alpha)

    def fit(self, train, metadata=None):
        self._fit_means(train)
        # Skip components that were already fitted (e.g. the same svd/content
        # instances are evaluated standalone and shared with the hybrid), so the
        # underlying models are not refitted. Unfitted components fit normally.
        if not self.cf.is_fitted:
            self.cf.fit(train, metadata)
        if not self.content.is_fitted:
            self.content.fit(train, metadata)
        return self

    def predict(self, user_id, parent_asin) -> float:
        cf_score = self.cf.predict(user_id, parent_asin)
        content_score = self.content.predict(user_id, parent_asin)
        blended = self.alpha * cf_score + (1.0 - self.alpha) * content_score
        return self._clip(blended)
