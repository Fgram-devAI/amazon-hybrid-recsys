"""End-to-end orchestrator test on a tmp processed dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.graph.analyze import run_graph_analysis


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    train = pd.DataFrame(
        [
            {"user_id": u, "parent_asin": it, "rating": 5.0, "timestamp": idx}
            for idx, (u, it) in enumerate(
                [
                    ("uA1", "iA1"), ("uA1", "iA2"), ("uA1", "iA3"),
                    ("uA2", "iA1"), ("uA2", "iA2"), ("uA2", "iA3"),
                    ("uA3", "iA1"), ("uA3", "iA2"), ("uA3", "iA3"),
                    ("uA4", "iA1"), ("uA4", "iA2"), ("uA4", "iA3"),
                    ("uB1", "iB1"), ("uB1", "iB2"), ("uB1", "iB3"),
                    ("uB2", "iB1"), ("uB2", "iB2"), ("uB2", "iB3"),
                    ("uB3", "iB1"), ("uB3", "iB2"), ("uB3", "iB3"),
                    ("uB4", "iB1"), ("uB4", "iB2"), ("uB4", "iB3"),
                    ("uX",  "iA1"), ("uX",  "iB1"),
                ]
            )
        ]
    )
    processed = tmp_path / "processed" / "toy"
    processed.mkdir(parents=True)
    train.to_parquet(processed / "train.parquet")
    # Optional explicit item -> category labels (used by alignment in this tiny test).
    # Real runs should derive labels from metadata.parquet when this file is absent.
    labels = pd.DataFrame(
        [{"parent_asin": it, "category": cat} for it, cat in [
            ("iA1", "X"), ("iA2", "X"), ("iA3", "X"),
            ("iB1", "Y"), ("iB2", "Y"), ("iB3", "Y"),
        ]]
    )
    labels.to_parquet(processed / "item_labels.parquet")
    return processed


def test_orchestrator_writes_report_with_expected_keys(tmp_dataset: Path) -> None:
    config = {
        "graph_analysis": {
            "min_shared_users": 2,
            "projection_weight": "weight_jaccard",
            "top_n_items": None,
            "spectral_k_values": [2],
            "girvan_newman_max_nodes": 500,
            "random_state": 42,
        }
    }
    report = run_graph_analysis(
        processed_dir=tmp_dataset,
        dataset="toy",
        config=config,
    )

    out_path = tmp_dataset / "graph_analysis" / "report.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk == report

    assert "bipartite_eda" in report
    assert "projection_eda" in report
    assert "communities" in report
    assert "louvain" in report["communities"]
    assert "spectral_k=2" in report["communities"]
    assert "girvan_newman" in report["communities"]
    # Leiden key is always present (None when leidenalg is missing).
    assert "leiden" in report["communities"]
    # Each present method reports a partition + alignment block.
    louvain = report["communities"]["louvain"]
    assert louvain["n_communities"] == 2
    assert "alignment" in louvain
    assert louvain["alignment"]["purity"] == pytest.approx(1.0)


def test_orchestrator_supports_overrides_and_custom_output(tmp_dataset: Path) -> None:
    config = {
        "graph_analysis": {
            "min_shared_users": 99,
            "projection_weight": "weight_jaccard",
            "top_n_items": None,
            "spectral_k_values": [99],
            "girvan_newman_max_nodes": 500,
            "random_state": 42,
        }
    }
    report = run_graph_analysis(
        processed_dir=tmp_dataset,
        dataset="toy",
        config=config,
        overrides={
            "min_shared_users": 2,
            "top_n_items": 4,
            "spectral_k_values": [2],
        },
        output_name="report_min2_top4.json",
    )

    out_path = tmp_dataset / "graph_analysis" / "report_min2_top4.json"
    assert out_path.exists()
    assert report["graph_analysis_config"]["min_shared_users"] == 2
    assert report["graph_analysis_config"]["top_n_items"] == 4
    assert report["graph_analysis_config"]["spectral_k_values"] == [2]
    assert report["projection_eda"]["n_nodes"] == 4
