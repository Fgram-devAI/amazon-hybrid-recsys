"""Tests for the evaluation runner (synthetic models + tiny data)."""

import pandas as pd

from src.evaluation.evaluate import evaluate_models, sample_negatives
from src.models.base import Recommender

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i2", "rating": 4.0},
    ]
)
TEST = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i3", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i4", "rating": 2.0},
    ]
)
META = pd.DataFrame(
    {"parent_asin": ["i1", "i2", "i3", "i4", "i5"], "text": ["a", "b", "c", "d", "e"]}
)


class FixedScore(Recommender):
    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return 5.0  # ranks everything equally; smoke-tests the harness

    def save_checkpoint(self, path):
        path.write_text("checkpoint")


def test_sample_negatives_is_reproducible_and_excludes_exclude_set():
    import numpy as np

    items = ["i1", "i2", "i3", "i4", "i5"]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = sample_negatives(items, exclude={"i1"}, n=2, rng=rng1)
    b = sample_negatives(items, exclude={"i1"}, n=2, rng=rng2)
    assert a == b                      # reproducible under same seed
    assert "i1" not in a               # excludes seen/positive items


def test_evaluate_models_returns_one_row_per_model_with_metric_columns():
    table = evaluate_models(
        {"fixed": FixedScore()},
        TRAIN,
        TEST,
        META,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
    )
    assert list(table["model"]) == ["fixed"]
    for col in ["dataset", "model", "rmse", "mae",
                "precision_at_k", "recall_at_k", "f1_at_k"]:
        assert col in table.columns
    assert table["rmse"].iloc[0] >= 0.0


def test_evaluate_models_can_cap_ranking_users():
    test = pd.DataFrame(
        [
            {"user_id": f"u{i}", "parent_asin": f"t{i}", "rating": 5.0}
            for i in range(5)
        ]
    )
    train = pd.DataFrame(
        [
            {"user_id": f"u{i}", "parent_asin": f"i{i}", "rating": 4.0}
            for i in range(5)
        ]
    )
    metadata = pd.DataFrame({"parent_asin": [f"i{i}" for i in range(5)]})

    table = evaluate_models(
        {"fixed": FixedScore()},
        train,
        test,
        metadata,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
        max_eval_users=2,
    )

    assert table["n_eval_users"].iloc[0] == 2
    assert table["max_eval_users"].iloc[0] == 2


def test_evaluate_models_can_cap_rating_rows():
    table = evaluate_models(
        {"fixed": FixedScore()},
        TRAIN,
        pd.concat([TEST, TEST, TEST], ignore_index=True),
        META,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
        max_test_rows=2,
    )

    assert table["max_test_rows"].iloc[0] == 2


def test_evaluate_models_checkpoints_graph_models(tmp_path):
    evaluate_models(
        {"lightgcn": FixedScore(), "graphsage": FixedScore()},
        TRAIN,
        TEST,
        META,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
        checkpoint_dir=tmp_path,
    )

    assert (tmp_path / "lightgcn.pt").read_text() == "checkpoint"
    assert (tmp_path / "graphsage.pt").read_text() == "checkpoint"


def test_evaluate_models_can_tag_graph_checkpoints(tmp_path):
    evaluate_models(
        {"lightgcn": FixedScore(), "graphsage": FixedScore()},
        TRAIN,
        TEST,
        META,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
        checkpoint_dir=tmp_path,
        checkpoint_tag="20ep",
    )

    assert (tmp_path / "lightgcn_20ep.pt").read_text() == "checkpoint"
    assert (tmp_path / "graphsage_20ep.pt").read_text() == "checkpoint"
    assert not (tmp_path / "lightgcn.pt").exists()
    assert not (tmp_path / "graphsage.pt").exists()


def test_build_models_shares_hybrid_component_instances():
    from src.evaluation.evaluate import build_models
    from src.models.embedding import FakeEmbedder

    config = {
        "processed_dir": "data/processed",
        "hybrid": {"alpha": 0.5},
        "models": {"ranking_random_seed": 42},
    }
    models = build_models(config, "tiny", FakeEmbedder(dim=8), no_knn=True)

    # the hybrid reuses the standalone svd/content instances -> fitted once each
    assert models["hybrid"].cf is models["svd"]
    assert models["hybrid"].content is models["content"]


def test_sample_negatives_returns_available_when_exclude_exceeds_catalog():
    import numpy as np

    items = np.asarray(["i1", "i2", "i3"], dtype=object)
    # exclude = 1 catalog item + 3 out-of-catalog positives -> len(exclude) > catalog
    exclude = {"i1", "p1", "p2", "p3"}
    negs = sample_negatives(items, exclude, n=2, rng=np.random.default_rng(42))
    assert set(negs) == {"i2", "i3"}  # available catalog negatives, not []


def test_evaluate_cli_accepts_include_ablation_flag(monkeypatch, tmp_path):
    from src.evaluation import evaluate as ev

    captured: dict = {}

    def fake_build_models(*args, **kwargs):
        captured["include_ablation"] = kwargs.get("include_ablation")
        return {}

    def fake_evaluate_models(*args, **kwargs):
        import pandas as pd
        return pd.DataFrame([{"model": "stub", "rmse": 0.0, "mae": 0.0}])

    def fake_load_processed(processed_dir, dataset):
        import pandas as pd
        return (pd.DataFrame(columns=["user_id", "parent_asin", "rating"]),
                pd.DataFrame(columns=["user_id", "parent_asin", "rating"]),
                pd.DataFrame(columns=["parent_asin"]))

    monkeypatch.setattr(ev, "build_models", fake_build_models)
    monkeypatch.setattr(ev, "evaluate_models", fake_evaluate_models)
    monkeypatch.setattr(ev, "_load_processed", fake_load_processed)
    monkeypatch.setattr("src.data.config.load_config", lambda _p: {
        "processed_dir": str(tmp_path),
        "models": {"ranking_random_seed": 42},
        "hybrid": {"alpha": 0.5},
        "evaluation": {"k": 10},
        "preprocessing": {"min_rating_relevant": 4.0},
        "advanced_features": {},
    })
    monkeypatch.setattr("src.models.embedding.build_embedder", lambda _c: object())

    ev.main(["--dataset", "tiny", "--no-knn", "--advanced", "--include-ablation", "--quiet"])
    assert captured["include_ablation"] is True


def test_include_ablation_without_advanced_is_rejected():
    import pytest
    from src.evaluation import evaluate as ev
    with pytest.raises(SystemExit):
        ev.main(["--dataset", "tiny", "--no-knn", "--include-ablation", "--quiet"])


def test_evaluate_cli_accepts_graph_only_flag(monkeypatch, tmp_path):
    from src.evaluation import evaluate as ev

    captured: dict = {}

    def fake_build_models(*args, **kwargs):
        captured["graph"] = kwargs.get("graph")
        captured["graph_only"] = kwargs.get("graph_only")
        return {}

    def fake_evaluate_models(*args, **kwargs):
        return pd.DataFrame([{"model": "stub", "rmse": 0.0, "mae": 0.0}])

    def fake_load_processed(processed_dir, dataset):
        return (
            pd.DataFrame(columns=["user_id", "parent_asin", "rating"]),
            pd.DataFrame(columns=["user_id", "parent_asin", "rating"]),
            pd.DataFrame(columns=["parent_asin"]),
        )

    monkeypatch.setattr(ev, "build_models", fake_build_models)
    monkeypatch.setattr(ev, "evaluate_models", fake_evaluate_models)
    monkeypatch.setattr(ev, "_load_processed", fake_load_processed)
    monkeypatch.setattr("src.data.config.load_config", lambda _p: {
        "processed_dir": str(tmp_path),
        "models": {"ranking_random_seed": 42},
        "hybrid": {"alpha": 0.5},
        "evaluation": {"k": 10},
        "preprocessing": {"min_rating_relevant": 4.0},
    })
    monkeypatch.setattr("src.models.embedding.build_embedder", lambda _c: object())

    ev.main(["--dataset", "tiny", "--graph-only", "--quiet"])
    assert captured == {"graph": True, "graph_only": True}


def test_graph_only_rejects_advanced_and_tune_alpha():
    import pytest
    from src.evaluation import evaluate as ev

    with pytest.raises(SystemExit):
        ev.main(["--dataset", "tiny", "--graph-only", "--advanced", "--quiet"])
    with pytest.raises(SystemExit):
        ev.main(["--dataset", "tiny", "--graph-only", "--tune-alpha", "--quiet"])


def test_checkpoint_tag_requires_graph():
    import pytest
    from src.evaluation import evaluate as ev

    with pytest.raises(SystemExit):
        ev.main(["--dataset", "tiny", "--checkpoint-tag", "20ep", "--quiet"])
