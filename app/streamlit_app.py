"""Streamlit dashboard entrypoint for the Amazon Hybrid RecSys project.

Launch with::

    streamlit run app/streamlit_app.py

The script reads ``app/assets/demo/`` by default and switches to
``data/processed/<dataset>/`` when those artifacts are present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import charts  # noqa: E402
from app.data_loader import DashboardData, load_dashboard_data  # noqa: E402

try:  # noqa: SIM105 - optional dependency path; app still runs without dotenv.
    from dotenv import load_dotenv  # noqa: E402
except ImportError:  # pragma: no cover - requirements includes it for full setup.
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


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
        "Recommendations",
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
    with tabs[5]:
        _render_recommendations(data, processed_dir=processed_dir)


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
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.set_index("Model")[["RMSE", "MAE"]])
        if "F1@10" in df.columns:
            st.bar_chart(df.set_index("Model")[["P@10", "R@10", "F1@10"]])

    if "alpha_sweep" in tables:
        st.subheader("Calibrated hybrid α sweep")
        df = charts.metrics_table(tables["alpha_sweep"], label_col="alpha")
        st.dataframe(df, hide_index=True, width="stretch")

    if "sentiment_ablation" in tables:
        st.subheader("Sentiment ablation")
        df = charts.metrics_table(tables["sentiment_ablation"], label_col="model")
        st.dataframe(df, hide_index=True, width="stretch")

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
        st.dataframe(df, hide_index=True, width="stretch")

    if "graphsage_bpr_ablation" in tables:
        st.subheader("GraphSAGE-BPR feature ablation")
        df = charts.metrics_table(
            tables["graphsage_bpr_ablation"],
            label_col="feature_set",
        )
        st.dataframe(df, hide_index=True, width="stretch")

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
        st.dataframe(df, hide_index=True, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(
        "Demo mode shows a small curated metadata sample. With local "
        "`data/processed/<dataset>/sample_items.json` present, the local "
        "list is used instead."
    )


RECOMMENDATION_METHOD_LABELS = (
    "Popularity",
    "SVD",
    "Content enriched",
    "Calibrated hybrid",
    "Item-KNN cosine",
    "Item-KNN pearson",
    "Item-KNN msd",
    "LightGCN checkpoint",
    "GraphSAGE MSE checkpoint",
    "LLM hybrid — profile mode",
    "LLM hybrid — query mode",
)


class _PreFitWrapper:
    """Wrap a cached, already-fitted model so re-calls of ``.fit`` are no-ops.

    The runner functions in ``app.recommender_runner`` always call
    ``model = factory(); model.fit(train)``. We cache the fitted model with
    ``@st.cache_resource`` and pass a factory that returns this wrapper, so the
    second call's ``.fit(train)`` does not re-train.
    """

    def __init__(self, model):
        self._model = model

    def fit(self, train):  # noqa: ARG002 - signature compat with Recommender
        return self

    def predict(self, user_id, parent_asin):
        return self._model.predict(user_id, parent_asin)


@st.cache_data(show_spinner=False)
def _cached_local_artifacts(processed_dir_str: str, dataset: str):
    from app.recommender_runner import load_local_artifacts

    return load_local_artifacts(processed_dir=Path(processed_dir_str), dataset=dataset)


@st.cache_resource(show_spinner=False)
def _cached_fitted_svd(processed_dir_str: str, dataset: str):
    """Fit SVD once per dataset; cache the model itself (not per-user rows)."""
    from src.models.cf import SVDRecommender

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    if not artifacts.available:
        return None
    model = SVDRecommender(n_factors=32, n_epochs=10, random_state=42)
    model.fit(artifacts.train)
    return model


@st.cache_resource(show_spinner=False)
def _cached_fitted_knn(processed_dir_str: str, dataset: str, sim_name: str):
    """Fit Item-KNN once per (dataset, sim_name); cache the model itself."""
    from src.models.cf import KNNRecommender

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    if not artifacts.available:
        return None
    model = KNNRecommender(k=40, sim_name=sim_name, user_based=False)
    model.fit(artifacts.train)
    return model


@st.cache_resource(show_spinner=False)
def _cached_fitted_content_enriched(processed_dir_str: str, dataset: str):
    """Fit the enriched content model once per dataset."""
    from src.data.config import load_config
    from src.models.content_enriched import ContentEnrichedRecommender
    from src.models.embedding import build_embedder

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    if not artifacts.available:
        return None
    config = load_config("config/config.yaml")
    advanced = config.get("advanced_features", {})
    embedder = build_embedder(config)
    model = ContentEnrichedRecommender(
        embedder,
        generic_roots=advanced.get("generic_category_roots", []),
        max_vocab=int(advanced.get("category_vocab_max", 256)),
        min_doc_freq=int(advanced.get("category_min_doc_freq", 5)),
        cache_dir=Path(processed_dir_str) / dataset / "advanced_features" / "title_desc_embeddings",
        review_features_dir=Path(processed_dir_str) / dataset / "advanced_features",
    )
    model.fit(artifacts.train, artifacts.metadata)
    return model


@st.cache_resource(show_spinner=False)
def _cached_fitted_calibrated_hybrid(processed_dir_str: str, dataset: str):
    """Fit calibrated hybrid once per dataset for qualitative comparison."""
    from src.data.config import load_config
    from src.models.calibrated_hybrid import CalibratedHybrid
    from src.models.cf import SVDRecommender

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    if not artifacts.available:
        return None
    config = load_config("config/config.yaml")
    content = _cached_fitted_content_enriched(processed_dir_str, dataset)
    if content is None:
        return None
    tuning = config.get("hybrid", {}).get("tuning", {})
    model = CalibratedHybrid(
        SVDRecommender(n_factors=32, n_epochs=10, random_state=42),
        content,
        alpha=float(config.get("hybrid", {}).get("alpha", 0.5)),
        calibration_max_rows=tuning.get("calibration_max_rows"),
        progress=False,
    )
    model.fit(artifacts.train, artifacts.metadata)
    return model


@st.cache_resource(show_spinner=False)
def _cached_lightgcn_scorer(processed_dir_str: str, dataset: str, checkpoint_str: str):
    """Load the LightGCN scorer once per checkpoint file; cache the callable."""
    from src.reasoning.candidates import load_lightgcn_from_checkpoint

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    ckpt = Path(checkpoint_str)
    if not artifacts.available or not ckpt.is_file():
        return None
    try:
        return load_lightgcn_from_checkpoint(ckpt, artifacts.train, {})
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_graphsage_model(processed_dir_str: str, dataset: str, checkpoint_str: str):
    """Load a GraphSAGE MSE checkpoint once per dataset/checkpoint."""
    from src.data.config import load_config
    from src.models.embedding import build_embedder
    from src.models.graphsage import GraphSAGERecommender

    artifacts = _cached_local_artifacts(processed_dir_str, dataset)
    ckpt = Path(checkpoint_str)
    if not artifacts.available or not ckpt.is_file():
        return None
    config = load_config("config/config.yaml")
    graph = config.get("graph", {})
    advanced = config.get("advanced_features", {})
    embedder = build_embedder(config)
    model = GraphSAGERecommender(
        embedder=embedder,
        generic_roots=advanced.get("generic_category_roots", []),
        max_vocab=int(advanced.get("category_vocab_max", 256)),
        min_doc_freq=int(advanced.get("category_min_doc_freq", 5)),
        hidden_dim=int(graph.get("embedding_dim", 64)),
        n_layers=int(graph.get("n_layers", 2)),
        epochs=0,
        lr=float(graph.get("lr", 0.005)),
        weight_decay=float(graph.get("weight_decay", 0.0)),
        batch_size=int(graph.get("batch_size", 1024)),
        seed=int(graph.get("seed", 42)),
        device=str(graph.get("device", "auto")),
        cache_dir=Path(processed_dir_str) / dataset / "advanced_features" / "title_desc_embeddings",
        review_features_dir=Path(processed_dir_str) / dataset / "advanced_features",
        progress=False,
    )
    try:
        model.prepare_for_checkpoint(artifacts.train, artifacts.metadata)
        model.load_checkpoint(ckpt)
    except Exception:
        return None
    return model


def _first_existing_checkpoint(base: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return base / names[0]


def _render_recommendations(data, *, processed_dir: Path) -> None:
    import pandas as pd

    from app.recommender_runner import (
        representative_users,
        run_knn,
        run_lightgcn_checkpoint,
        run_llm_hybrid,
        run_popularity,
        run_svd,
        seen_items_for_user,
        user_history_rows,
    )

    st.header("Recommendations")
    st.caption(
        "Qualitative inspection of recommendations for a single train user. "
        "Outputs here are not a replacement for the measured benchmark metrics."
    )

    # Demo mode: never run models.
    if data.mode != "local":
        st.warning(
            "Interactive recommendations require local processed artifacts. "
            "Run preprocessing/model/storage commands from README first."
        )
        return

    artifacts = _cached_local_artifacts(str(processed_dir), data.dataset)
    if not artifacts.available:
        st.warning(
            "Interactive recommendations require local processed artifacts. "
            "Run preprocessing/model/storage commands from README first."
        )
        for reason in artifacts.missing_reasons:
            st.caption(reason)
        return

    left, main = st.columns([1, 2])

    with left:
        st.subheader("Controls")
        users = representative_users(artifacts.train, limit=50)
        if not users:
            st.warning("No users with enough interactions in the local train split.")
            return
        user_id = st.selectbox("User", users, index=0)
        method = st.selectbox("Method", RECOMMENDATION_METHOD_LABELS, index=0)
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        candidate_pool_size = st.slider(
            "Candidate pool",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help=(
                "SVD, KNN, content, hybrid, and graph checkpoints score this "
                "many unseen candidates before taking Top-K. Larger pools are "
                "slower but reduce popularity-pool bias."
            ),
        )
        seed_asin = st.text_input("Optional seed parent_asin", value="").strip()
        free_text_query = st.text_input(
            "Optional query (LLM query mode only)", value=""
        ).strip()
        live_llm_requested = st.checkbox(
            "Call Groq for explanation (live LLM, costs tokens)",
            value=False,
            help="Off by default. When off, LLM modes return the dry-run evidence payload.",
        )
        live_llm = live_llm_requested
        if live_llm_requested and not os.environ.get("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. Falling back to dry-run.")
            live_llm = False
        run_clicked = st.button("Run", type="primary")

    with main:
        st.subheader("Selected user history")
        history = user_history_rows(
            artifacts.train,
            artifacts.metadata,
            user_id=user_id,
            min_rating=4.0,
            limit=10,
        )
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        else:
            st.caption("No train items with rating ≥ 4 for this user.")

        if seed_asin:
            meta = artifacts.metadata
            match = meta.loc[meta["parent_asin"].astype(str) == seed_asin]
            if not match.empty:
                row = match.iloc[0]
                st.caption(f"Seed item: {row.get('title', seed_asin)} ({seed_asin})")
            else:
                st.caption(f"Seed parent_asin {seed_asin} not found in metadata.")

        if not run_clicked:
            st.caption("Pick a method and press Run.")
            return

        rows: list[dict] = []
        warning: str | None = None
        raw_llm: dict | None = None

        if method == "Popularity":
            rows = run_popularity(
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                seed_asin=seed_asin or None,
            )
        elif method == "SVD":
            cached_svd = _cached_fitted_svd(str(processed_dir), data.dataset)
            if cached_svd is None:
                st.info("SVD unavailable.")
                return
            rows = run_svd(
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                seed_asin=seed_asin or None,
                svd_factory=lambda: _PreFitWrapper(cached_svd),
            )
        elif method == "Content enriched":
            from app.recommender_runner import run_fitted_recommender

            cached_content = _cached_fitted_content_enriched(str(processed_dir), data.dataset)
            if cached_content is None:
                st.info("Content enriched unavailable.")
                return
            rows = run_fitted_recommender(
                model=cached_content,
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                method="content_enriched",
                score_field="semantic_score",
                score_sources=["content_enriched"],
                candidate_pool_size=candidate_pool_size,
                seed_asin=seed_asin or None,
            )
        elif method == "Calibrated hybrid":
            from app.recommender_runner import run_fitted_recommender

            cached_hybrid = _cached_fitted_calibrated_hybrid(str(processed_dir), data.dataset)
            if cached_hybrid is None:
                st.info("Calibrated hybrid unavailable.")
                return
            rows = run_fitted_recommender(
                model=cached_hybrid,
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                method="calibrated_hybrid",
                score_field="hybrid_score",
                score_sources=["svd", "content_enriched"],
                candidate_pool_size=candidate_pool_size,
                seed_asin=seed_asin or None,
            )
        elif method.startswith("Item-KNN"):
            sim_name = method.split()[-1]
            cached_knn = _cached_fitted_knn(str(processed_dir), data.dataset, sim_name)
            if cached_knn is None:
                st.info(f"Item-KNN ({sim_name}) unavailable.")
                return
            rows = run_knn(
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                sim_name=sim_name,
                candidate_pool_size=candidate_pool_size,
                seed_asin=seed_asin or None,
                knn_factory=lambda _name: _PreFitWrapper(cached_knn),
            )
        elif method == "LightGCN checkpoint":
            ckpt = (
                processed_dir / data.dataset / "graph_checkpoints"
                / "lightgcn_40ep_neg4_wd1e-5.pt"
            )
            cached_scorer = _cached_lightgcn_scorer(
                str(processed_dir), data.dataset, str(ckpt)
            )

            def _scorer_loader(_path, _train, _config):
                return cached_scorer

            if cached_scorer is None:
                # Fall back to the runner's own missing-file handling so the user
                # gets a helpful "checkpoint not found" warning.
                result = run_lightgcn_checkpoint(
                    train=artifacts.train,
                    metadata=artifacts.metadata,
                    user_id=user_id,
                    top_k=top_k,
                    checkpoint_path=ckpt,
                    loader=None,
                    config={},
                    candidate_pool_size=candidate_pool_size,
                    seed_asin=seed_asin or None,
                )
            else:
                result = run_lightgcn_checkpoint(
                    train=artifacts.train,
                    metadata=artifacts.metadata,
                    user_id=user_id,
                    top_k=top_k,
                    checkpoint_path=ckpt,
                    loader=_scorer_loader,
                    config={},
                    candidate_pool_size=candidate_pool_size,
                    seed_asin=seed_asin or None,
                )
            rows = result.rows
            warning = result.warning
        elif method == "GraphSAGE MSE checkpoint":
            from app.recommender_runner import run_fitted_recommender

            ckpt = _first_existing_checkpoint(
                processed_dir / data.dataset / "graph_checkpoints",
                ("graphsage_20ep.pt", "graphsage.pt"),
            )
            cached_graphsage = _cached_graphsage_model(str(processed_dir), data.dataset, str(ckpt))
            if cached_graphsage is None:
                st.warning(
                    f"GraphSAGE checkpoint not found or could not be loaded at {ckpt}."
                )
                return
            rows = run_fitted_recommender(
                model=cached_graphsage,
                train=artifacts.train,
                metadata=artifacts.metadata,
                user_id=user_id,
                top_k=top_k,
                method="graphsage_mse",
                score_field="hybrid_score",
                score_sources=["graphsage_mse"],
                candidate_pool_size=candidate_pool_size,
                seed_asin=seed_asin or None,
            )
        elif method.startswith("LLM hybrid"):
            mode = "profile" if "profile" in method else "query"
            llm_result = run_llm_hybrid(
                dataset=data.dataset,
                user_id=user_id,
                top_k=top_k,
                mode=mode,
                query=free_text_query or None,
                dry_run=not live_llm,
                metadata=artifacts.metadata,
                seen=seen_items_for_user(artifacts.train, user_id=user_id),
            )
            rows = llm_result.rows
            warning = llm_result.warning
            raw_llm = llm_result.raw_result

        seen = seen_items_for_user(artifacts.train, user_id=user_id)
        for row in rows:
            if row.get("already_seen") is None:
                row["already_seen"] = str(row.get("parent_asin")) in seen

        if warning:
            st.warning(warning)
        if not rows:
            st.info("No recommendations to display.")
            return

        st.subheader("Recommendations")
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")

        if raw_llm is not None:
            with st.expander("LLM evidence / prompt (full payload)"):
                st.json(raw_llm)


if __name__ == "__main__":
    render()
