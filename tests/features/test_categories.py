"""Tests for category normalization and filtering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.categories import (
    build_category_features,
    build_category_vocab,
    filter_categories,
    normalize_category,
    save_category_artifacts,
)


def test_normalize_strips_whitespace_and_case():
    assert normalize_category("  comedy  ") == "Comedy"
    assert normalize_category("ACTION & ADVENTURE") == "Action & Adventure"
    assert normalize_category("movies & tv") == "Movies & TV"
    assert normalize_category("rpg") == "RPG"
    assert normalize_category("dvd") == "DVD"


def test_normalize_returns_none_for_empty():
    assert normalize_category("") is None
    assert normalize_category(None) is None
    assert normalize_category("   ") is None


def test_filter_drops_generic_roots_and_empty_keeps_informative():
    raw = ["Movies & TV", "Comedy", "", None, "Drama", "movies & tv"]
    generic = ["Movies & TV", "Video Games", "Digital Music"]
    assert filter_categories(raw, generic_roots=generic) == ["Comedy", "Drama"]


def test_filter_is_order_preserving_and_deduplicated():
    raw = ["Comedy", "Drama", "Comedy", "Drama"]
    assert filter_categories(raw, generic_roots=[]) == ["Comedy", "Drama"]


_METADATA = pd.DataFrame(
    [
        {"parent_asin": "i1", "categories": ["Movies & TV", "Comedy", "Drama"]},
        {"parent_asin": "i2", "categories": ["Comedy"]},
        {"parent_asin": "i3", "categories": ["Drama", "Action & Adventure"]},
        {"parent_asin": "i4", "categories": []},
    ]
)


def test_build_vocab_filters_and_caps():
    vocab = build_category_vocab(
        _METADATA,
        generic_roots=["Movies & TV"],
        max_vocab=2,
        min_doc_freq=1,
    )
    # "Comedy" and "Drama" both appear twice -> they win the cap; ties broken by name.
    assert vocab == ["Comedy", "Drama"]


def test_build_vocab_respects_min_doc_freq():
    vocab = build_category_vocab(
        _METADATA, generic_roots=[], max_vocab=10, min_doc_freq=2
    )
    # Only categories appearing on >= 2 items survive.
    assert vocab == ["Comedy", "Drama"]


def test_build_features_returns_multi_hot_aligned_to_ids():
    vocab = ["Comedy", "Drama"]
    feats, ids = build_category_features(_METADATA, vocab, generic_roots=["Movies & TV"])
    assert ids == ["i1", "i2", "i3", "i4"]
    np.testing.assert_array_equal(
        feats,
        np.array(
            [
                [1, 1],  # i1 -> Comedy + Drama
                [1, 0],  # i2 -> Comedy
                [0, 1],  # i3 -> Drama (Action & Adventure not in vocab)
                [0, 0],  # i4 -> empty
            ],
            dtype=np.float32,
        ),
    )


def test_save_artifacts_writes_vocab_features_meta(tmp_path: Path):
    vocab = ["Comedy", "Drama"]
    feats, ids = build_category_features(_METADATA, vocab, generic_roots=["Movies & TV"])
    save_category_artifacts(
        out_dir=tmp_path,
        vocab=vocab,
        features=feats,
        ids=ids,
        dataset="tiny",
        meta_row_count=len(_METADATA),
    )
    assert json.loads((tmp_path / "category_vocab.json").read_text()) == vocab
    assert json.loads((tmp_path / "item_feature_ids.json").read_text()) == ids
    meta = json.loads((tmp_path / "item_features_v2_meta.json").read_text())
    assert meta["dataset"] == "tiny"
    assert meta["vocab_size"] == 2
    assert meta["row_count"] == 4
    assert np.load(tmp_path / "item_features_v2.npy").shape == (4, 2)
