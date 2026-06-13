"""Tests for the audit-shared helpers used by all three checkpoint evaluators."""

import json

import pytest

from src.evaluation._audit_shared import (
    compute_checkpoint_audit_metrics,
    processed_dataset_key,
    requested_split_protocol,
    resolve_split_protocol,
)


def test_resolve_split_protocol_reads_eda_summary(tmp_path):
    processed = tmp_path / "processed"
    (processed / "video_games").mkdir(parents=True)
    (processed / "video_games" / "eda_summary.json").write_text(
        json.dumps({"split_protocol": "leave_last_out"})
    )
    result = resolve_split_protocol(
        str(processed), "video_games", {"split_protocol": "per_user_chronological_80_20"}
    )
    assert result == "leave_last_out"


def test_resolve_split_protocol_reads_suffixed_leave_last_out_summary(tmp_path):
    processed = tmp_path / "processed"
    (processed / "video_games__leave_last_out").mkdir(parents=True)
    (processed / "video_games__leave_last_out" / "eda_summary.json").write_text(
        json.dumps({"split_protocol": "leave_last_out"})
    )
    result = resolve_split_protocol(
        str(processed), "video_games", {"split_protocol": "leave_last_out"}
    )
    assert result == "leave_last_out"


def test_processed_dataset_key_adds_non_default_suffix_once():
    assert processed_dataset_key("video_games", "per_user_chronological_80_20") == "video_games"
    assert processed_dataset_key("video_games", "leave_last_out") == "video_games__leave_last_out"
    assert (
        processed_dataset_key("video_games__leave_last_out", "leave_last_out")
        == "video_games__leave_last_out"
    )


def test_requested_split_protocol_prefers_non_default_preprocessing_split():
    config = {
        "preprocessing": {"split_protocol": "leave_last_out"},
        "evaluation": {"split_protocol": "per_user_chronological_80_20"},
    }
    assert requested_split_protocol(config) == "leave_last_out"


def test_requested_split_protocol_uses_evaluation_when_preprocessing_default():
    config = {
        "preprocessing": {"split_protocol": "per_user_chronological_80_20"},
        "evaluation": {"split_protocol": "leave_last_out"},
    }
    assert requested_split_protocol(config) == "leave_last_out"


def test_resolve_split_protocol_falls_back_to_config(tmp_path):
    processed = tmp_path / "processed"
    (processed / "video_games").mkdir(parents=True)
    # No eda_summary.json
    result = resolve_split_protocol(
        str(processed), "video_games", {"split_protocol": "per_user_chronological_80_20"}
    )
    assert result == "per_user_chronological_80_20"


def test_resolve_split_protocol_default_when_no_config():
    result = resolve_split_protocol("/nonexistent", "x", {})
    assert result == "per_user_chronological_80_20"


def test_compute_checkpoint_audit_metrics_shape_and_split_protocol():
    per_user_data = [
        {"ranked": ["a", "b", "c"], "relevant": {"a"}},
        {"ranked": ["a", "b", "c"], "relevant": {"b"}},
    ]
    metrics = compute_checkpoint_audit_metrics(
        per_user_data, k=3, split_protocol="leave_last_out"
    )
    for key in [
        "precision_at_k", "recall_at_k", "f1_at_k",
        "hit_rate_at_k", "ndcg_at_k",
        "oracle_precision_at_k", "oracle_recall_at_k", "oracle_f1_at_k",
        "oracle_hit_rate_at_k", "oracle_ndcg_at_k",
        "precision_oracle_ratio_at_k", "recall_oracle_ratio_at_k",
        "f1_oracle_ratio_at_k",
        "split_protocol", "n_eval_users", "k",
    ]:
        assert key in metrics, f"missing key {key}"
    assert metrics["split_protocol"] == "leave_last_out"
    assert metrics["n_eval_users"] == 2
    assert metrics["k"] == 3
    # Hand-computed: both users have P=1/3 (one hit in top-3); NDCG differs by rank
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)


def test_compute_checkpoint_audit_metrics_skips_users_with_no_relevant():
    per_user_data = [
        {"ranked": ["a", "b"], "relevant": set()},   # skipped
        {"ranked": ["a", "b"], "relevant": {"a"}},
    ]
    metrics = compute_checkpoint_audit_metrics(
        per_user_data, k=2, split_protocol="per_user_chronological_80_20"
    )
    assert metrics["n_eval_users"] == 1
