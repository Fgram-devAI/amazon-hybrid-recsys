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
