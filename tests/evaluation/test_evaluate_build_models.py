"""build_models should register all advanced models behind flags."""

from src.evaluation.evaluate import build_models
from src.models.embedding import FakeEmbedder


def _config():
    return {
        "processed_dir": "data/processed",
        "models": {"ranking_random_seed": 42},
        "hybrid": {"alpha": 0.5},
        "advanced_features": {
            "generic_category_roots": ["Movies & TV"],
            "category_vocab_max": 16,
            "category_min_doc_freq": 1,
        },
        "graph": {
            "embedding_dim": 4,
            "n_layers": 1,
            "epochs": 1,
            "lr": 0.05,
            "batch_size": 4,
            "num_negatives": 1,
            "device": "cpu",
            "seed": 0,
            "min_rating_positive": 4.0,
            "validation_fraction": 0.1,
        },
    }


def test_build_models_default_set_matches_phase1():
    models = build_models(_config(), "tiny", FakeEmbedder(dim=8), no_knn=True)
    assert set(models) == {"content", "svd", "hybrid"}


def test_build_models_with_advanced_flag_adds_baselines_and_enriched():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8), no_knn=True, advanced=True
    )
    assert {"random", "popularity", "content_enriched", "calibrated_hybrid"} <= set(
        models
    )


def test_calibrated_hybrid_shares_components_with_standalone_models():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8), no_knn=True, advanced=True
    )
    calibrated = models["calibrated_hybrid"]
    assert calibrated.cf is models["svd"]
    assert calibrated.content is models["content_enriched"]


def test_build_models_uses_explicit_alpha_override():
    models = build_models(_config(), "tiny", FakeEmbedder(dim=8), no_knn=True, alpha=0.75)
    assert models["hybrid"].alpha == 0.75


def test_build_models_with_ablation_registers_both_variants():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8),
        no_knn=True, advanced=True, include_ablation=True,
    )
    assert "content_enriched_with_sentiment" in models
    assert "content_enriched_no_sentiment" in models
    # In ablation mode we drop the legacy alias to avoid fitting the same model twice.
    assert "content_enriched" not in models


def test_ablation_no_sentiment_variant_has_flags_disabled():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8),
        no_knn=True, advanced=True, include_ablation=True,
    )
    no_sent = models["content_enriched_no_sentiment"]
    assert no_sent.use_item_sentiment is False
    assert no_sent.use_user_offset is False


def test_ablation_variants_share_embedder_and_cache_dir():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8),
        no_knn=True, advanced=True, include_ablation=True,
    )
    with_sent = models["content_enriched_with_sentiment"]
    no_sent = models["content_enriched_no_sentiment"]
    # Same embedder instance and same on-disk cache_dir -> apples-to-apples comparison.
    assert with_sent.embedder is no_sent.embedder
    assert str(with_sent.cache_dir) == str(no_sent.cache_dir)


def test_build_models_without_ablation_omits_ablation_variants():
    models = build_models(
        _config(), "tiny", FakeEmbedder(dim=8),
        no_knn=True, advanced=True, include_ablation=False,
    )
    assert "content_enriched_no_sentiment" not in models
    assert "content_enriched_with_sentiment" not in models
    # Backward-compat alias still present under the legacy name.
    assert "content_enriched" in models


def test_build_models_with_graph_flag_registers_graph_models():
    from src.evaluation.evaluate import build_models

    config = {
        "processed_dir": "data/processed",
        "models": {"ranking_random_seed": 42},
        "advanced_features": {},
        "hybrid": {"alpha": 0.5},
        "graph": {
            "embedding_dim": 4, "n_layers": 1, "epochs": 1, "lr": 0.05,
            "batch_size": 4, "num_negatives": 1, "device": "cpu", "seed": 0,
            "min_rating_positive": 4.0, "validation_fraction": 0.1,
        },
    }

    class _FakeEmbedder:
        name = "fake"
        device = "cpu"
        def encode(self, texts):
            import numpy as np
            return np.zeros((len(texts), 4), dtype="float32")

    models = build_models(
        config, dataset="ds", embedder=_FakeEmbedder(),
        no_knn=True, advanced=False, graph=True,
    )
    assert "lightgcn" in models
    assert "graphsage" in models
    assert "graphsage_bpr" in models


def test_build_models_graph_only_registers_only_graph_models():
    models = build_models(
        _config(),
        "tiny",
        FakeEmbedder(dim=8),
        no_knn=True,
        advanced=False,
        graph=True,
        graph_only=True,
    )
    assert set(models) == {"lightgcn", "graphsage", "graphsage_bpr"}


def test_build_models_graph_only_ignores_advanced_model_registration():
    models = build_models(
        _config(),
        "tiny",
        FakeEmbedder(dim=8),
        no_knn=True,
        advanced=True,
        graph=True,
        graph_only=True,
    )
    assert set(models) == {"lightgcn", "graphsage", "graphsage_bpr"}
