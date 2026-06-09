"""Collaborative-filtering recommenders backed by scikit-surprise.

SVD is the primary full-scale CF baseline. Item-KNN is provided too, but its
item-item similarity matrix is O(n_items^2) memory, so the caller must cap or
subsample items on large datasets (see evaluate.py guardrails).

scikit-surprise runs on CPU; CUDA/MPS acceleration is not available for these
baselines without replacing the backend.
"""

from surprise import SVD, Dataset, KNNWithMeans, Reader

from .base import Recommender


class _SurpriseRecommender(Recommender):
    def __init__(self, algo):
        super().__init__()
        self._algo = algo
        self.device = "cpu"

    def fit(self, train, metadata=None):
        self._fit_means(train)
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(
            train[["user_id", "parent_asin", "rating"]], reader
        )
        self._algo.fit(data.build_full_trainset())
        return self

    def predict(self, user_id, parent_asin) -> float:
        try:
            est = self._algo.predict(str(user_id), str(parent_asin)).est
        except Exception:
            est = self._fallback(user_id, parent_asin)
        return self._clip(est)


class SVDRecommender(_SurpriseRecommender):
    def __init__(self, **kwargs):
        super().__init__(SVD(**kwargs))


class KNNRecommender(_SurpriseRecommender):
    def __init__(self, k=40, **kwargs):
        super().__init__(
            KNNWithMeans(k=k, sim_options={"user_based": False}, verbose=False, **kwargs)
        )
