"""Tests for app.data_loader: demo fallback and local-prefer behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data_loader import DashboardData, load_dashboard_data


REPO_DEMO_DIR = Path(__file__).resolve().parents[2] / "app" / "assets" / "demo"


def test_demo_fallback_when_processed_dir_missing(tmp_path: Path) -> None:
    """When data/processed/<dataset> is absent, the loader uses the bundled demo bundle."""
    result = load_dashboard_data(
        "video_games",
        processed_dir=tmp_path / "does_not_exist",
        demo_dir=REPO_DEMO_DIR,
    )
    assert isinstance(result, DashboardData)
    assert result.mode == "demo"
    assert result.dataset == "video_games"
    assert result.eda_summary["after_kcore"] == 814586
    assert isinstance(result.model_metrics["tables"]["advanced"], list)
    assert result.graph_analysis is not None
    assert result.graph_subgraph_3d is not None
    assert len(result.sample_items) >= 10


def test_local_eda_takes_precedence(tmp_path: Path) -> None:
    """When local eda_summary.json exists, the loader reads it instead of the demo file."""
    local = tmp_path / "video_games"
    local.mkdir()
    local_eda = {
        "raw_reviews": 1,
        "after_drop_invalid": 1,
        "after_dedup": 1,
        "after_kcore": 1,
        "users_after": 1,
        "items_after": 1,
        "train_interactions": 1,
        "test_interactions": 1,
        "rating_hist_after": {"5.0": 1},
        "pct_relevant_after": 1.0,
        "k_core_applied": 5,
    }
    (local / "eda_summary.json").write_text(json.dumps(local_eda))

    result = load_dashboard_data(
        "video_games",
        processed_dir=tmp_path,
        demo_dir=REPO_DEMO_DIR,
    )
    assert result.mode == "local"
    assert result.eda_summary["after_kcore"] == 1
    assert result.model_metrics["tables"]["advanced"][0]["model"] == "content"
    notes_text = " ".join(result.notes)
    assert "model_metrics" in notes_text


def test_missing_demo_graph_summary_returns_none(tmp_path: Path) -> None:
    """If graph summaries are missing, graph fields are None."""
    fake_demo = tmp_path / "demo"
    fake_demo.mkdir()
    (fake_demo / "video_games_eda_summary.json").write_text(
        json.dumps({"k_core_applied": 5, "after_kcore": 0})
    )
    (fake_demo / "model_metrics_summary.json").write_text(json.dumps({"tables": {}}))
    (fake_demo / "sample_items.json").write_text(json.dumps([]))

    result = load_dashboard_data(
        "video_games",
        processed_dir=tmp_path / "no_local",
        demo_dir=fake_demo,
    )
    assert result.mode == "demo"
    assert result.graph_analysis is None
    assert result.graph_subgraph_3d is None


@pytest.mark.parametrize(
    "filename,required_top_level_key",
    [
        ("manifest.json", "dataset"),
        ("video_games_eda_summary.json", "after_kcore"),
        ("model_metrics_summary.json", "tables"),
        ("graph_analysis_summary.json", "projections"),
        ("graph_subgraph_3d.json", "nodes"),
    ],
)
def test_demo_bundle_has_required_keys(filename: str, required_top_level_key: str) -> None:
    """Spec section 6 schema check on the committed demo bundle."""
    payload = json.loads((REPO_DEMO_DIR / filename).read_text())
    assert required_top_level_key in payload


def test_demo_sample_items_have_required_fields() -> None:
    """Each sample item must expose the small fixed schema the UI consumes."""
    payload = json.loads((REPO_DEMO_DIR / "sample_items.json").read_text())
    assert isinstance(payload, list) and len(payload) >= 10
    required = {
        "parent_asin",
        "title",
        "display_category",
        "store",
        "price",
        "average_rating",
        "rating_number",
    }
    for item in payload:
        assert required.issubset(item.keys()), f"Missing keys in item: {set(item.keys())}"


def test_graph_subgraph_3d_is_capped_and_has_coordinates() -> None:
    """The committed graph sample must remain small enough for Streamlit demo mode."""
    payload = json.loads((REPO_DEMO_DIR / "graph_subgraph_3d.json").read_text())
    assert len(payload["nodes"]) <= 300
    assert len(payload["edges"]) <= 2000
    node_required = {"id", "label", "category", "community", "degree", "x", "y", "z"}
    edge_required = {"source", "target", "weight"}
    for node in payload["nodes"]:
        assert node_required.issubset(node.keys())
    for edge in payload["edges"]:
        assert edge_required.issubset(edge.keys())
