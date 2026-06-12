"""Streamlit dashboard entrypoint for the Amazon Hybrid RecSys project.

Launch with::

    streamlit run app/streamlit_app.py

The script reads ``app/assets/demo/`` by default and switches to
``data/processed/<dataset>/`` when those artifacts are present.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from app import charts
from app.data_loader import DashboardData, load_dashboard_data


def render() -> None:
    st.set_page_config(page_title="Amazon Hybrid RecSys", layout="wide")
    st.title("Amazon Hybrid Recommender System")
    st.write(
        "Hybrid recommender on Amazon Reviews 2023 (Video_Games benchmark). "
        "Content + collaborative filtering, plus graph (LightGCN/GraphSAGE) "
        "and feature-ablation diagnostics. Raw and processed Amazon data are "
        "reproducible local artifacts and are not committed."
    )

    processed_dir = Path(os.environ.get("RECSYS_PROCESSED_DIR", "data/processed"))
    demo_dir = Path(os.environ.get("RECSYS_DEMO_DIR", "app/assets/demo"))
    data = load_dashboard_data(processed_dir=processed_dir, demo_dir=demo_dir)
    _render_mode_banner(data)

    tabs = st.tabs([
        "Overview",
        "Model Comparison",
        "Graph Models + Ablations",
        "Graph EDA / Communities",
        "Item Explorer",
    ])

    with tabs[0]:
        _render_overview(data)
    with tabs[1]:
        _render_model_comparison(data)
    with tabs[2]:
        _render_graph_models(data)
    with tabs[3]:
        _render_graph_eda(data)
    with tabs[4]:
        _render_item_explorer(data)


def _render_mode_banner(data: DashboardData) -> None:
    if data.mode == "local":
        st.success("Mode: full local artifacts")
    else:
        st.info("Mode: bundled demo summaries")
    for note in data.notes:
        st.caption(note)


def _fmt_int(value: object) -> str:
    if isinstance(value, int | float):
        return f"{int(value):,}"
    return "-"


def _render_overview(data: DashboardData) -> None:
    eda = data.eda_summary
    st.header("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users (after k-core)", _fmt_int(eda.get("users_after")))
    col2.metric("Items (after k-core)", _fmt_int(eda.get("items_after")))
    col3.metric("Train interactions", _fmt_int(eda.get("train_interactions")))
    col4.metric("Test interactions", _fmt_int(eda.get("test_interactions")))

    st.subheader("Preprocessing funnel")
    funnel_df = charts.preprocessing_funnel(eda)
    st.bar_chart(funnel_df.set_index("step"))

    st.subheader("Rating distribution (after k-core)")
    if "rating_hist_after" in eda:
        hist_df = charts.rating_histogram(eda["rating_hist_after"])
        st.bar_chart(hist_df.set_index("rating"))

    st.subheader("Sparsity")
    st.write(
        f"Before k-core: {eda.get('sparsity_before', float('nan')):.6f} — "
        f"after k-core: {eda.get('sparsity_after', float('nan')):.6f} — "
        f"relevant ratings (rating ≥ 4): "
        f"{100 * eda.get('pct_relevant_after', 0):.2f}%"
    )
    st.caption(
        "Raw Amazon data is local and reproducible (`python -m src.data.fetch ...`) "
        "and is not committed to this repository."
    )


def _render_model_comparison(data: DashboardData) -> None:
    tables = data.model_metrics.get("tables", {})
    protocol = data.model_metrics.get(
        "ranking_protocol",
        "sampled candidates, K=10, 100 negatives/user, seed=42",
    )
    st.header("Model comparison")
    st.caption(f"Ranking protocol: {protocol}")

    if "advanced" in tables:
        st.subheader("Primary advanced-models table")
        df = charts.metrics_table(tables["advanced"], label_col="model")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Model")[["RMSE", "MAE"]])
        if "F1@10" in df.columns:
            st.bar_chart(df.set_index("Model")[["P@10", "R@10", "F1@10"]])

    if "alpha_sweep" in tables:
        st.subheader("Calibrated hybrid α sweep")
        df = charts.metrics_table(tables["alpha_sweep"], label_col="alpha")
        st.dataframe(df, hide_index=True, use_container_width=True)

    if "sentiment_ablation" in tables:
        st.subheader("Sentiment ablation")
        df = charts.metrics_table(tables["sentiment_ablation"], label_col="model")
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown(
        "**Current conclusions:** SVD is the strongest RMSE baseline; popularity "
        "and `content_enriched` are strong sampled-ranking baselines; "
        "`LightGCN 40ep / neg4 / wd1e-5` is the strongest graph ranker; "
        "`GraphSAGE-MSE 20ep` improves MAE/ranking but worsens RMSE vs 10ep."
    )


def _render_graph_models(data: DashboardData) -> None:
    tables = data.model_metrics.get("tables", {})
    st.header("Graph models + GraphSAGE-BPR feature ablation")

    if "graph" in tables:
        st.subheader("Graph checkpoint table")
        df = charts.metrics_table(tables["graph"], label_col="model")
        st.dataframe(df, hide_index=True, use_container_width=True)

    if "graphsage_bpr_ablation" in tables:
        st.subheader("GraphSAGE-BPR feature ablation")
        df = charts.metrics_table(
            tables["graphsage_bpr_ablation"],
            label_col="feature_set",
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown(
        "**Interpretation:** text embeddings carry the useful GraphSAGE-BPR "
        "feature signal; sentiment / user generosity is optional / noisy for "
        "GraphSAGE-BPR at this training budget; LightGCN remains the stronger "
        "graph ranker."
    )


def _render_graph_eda(data: DashboardData) -> None:
    st.header("Graph EDA / Communities")
    if data.graph_analysis is None:
        st.info("Graph analysis summary not available.")
        return

    st.subheader("Item-item projection comparison")
    rows = data.graph_analysis.get("projections", [])
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.subheader("Projection scale")
        st.bar_chart(df.set_index("projection")[["items", "edges", "largest_cc"]])
        alignment_cols = [
            c
            for c in ["louvain_purity", "louvain_nmi", "spectral_k50_nmi"]
            if c in df.columns
        ]
        if alignment_cols:
            st.subheader("Community/category alignment")
            st.line_chart(df.set_index("projection")[alignment_cols])

    if data.graph_subgraph_3d is not None:
        st.subheader("Largest Louvain community sample (3D)")
        st.caption(
            "Capped offline layout only: this is not the full item graph and no "
            "layout/community detection runs inside Streamlit."
        )
        fig = charts.graph_subgraph_3d_figure(data.graph_subgraph_3d)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("3D graph sample unavailable.")

    note = data.graph_analysis.get("girvan_newman_note")
    if note:
        st.subheader("Girvan-Newman")
        st.write(note)

    st.markdown(
        "Louvain is the most stable community method on the Video_Games "
        "projections; spectral clustering is weaker on category alignment. "
        "Stricter co-rating thresholds give smaller but cleaner communities, "
        "while broader projections improve catalog coverage at the cost of "
        "alignment quality."
    )


def _render_item_explorer(data: DashboardData) -> None:
    import pandas as pd

    st.header("Item Explorer")
    df = pd.DataFrame(data.sample_items)
    if df.empty:
        st.info("No sample items available.")
        return

    query = st.text_input("Search title / category / ASIN", value="")
    if query:
        mask = (
            df["title"].astype(str).str.contains(query, case=False, na=False)
            | df["display_category"].astype(str).str.contains(query, case=False, na=False)
            | df["parent_asin"].astype(str).str.contains(query, case=False, na=False)
        )
        df = df[mask]
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(
        "Demo mode shows a small curated metadata sample. With local "
        "`data/processed/<dataset>/sample_items.json` present, the local "
        "list is used instead."
    )


if __name__ == "__main__":
    render()
