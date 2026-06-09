# Amazon Hybrid Recommender System

A hybrid recommender system that combines **content-based filtering** and **collaborative filtering**, built on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. It compares the hybrid approach against standalone baselines on standard metrics and serves recommendations through a Streamlit app.

> MSc Artificial Intelligence — *Applications in AI*, Topic 2 (Hybrid Recommender System).

## Overview

Two recommendation paradigms, each with a known weakness:

- **Content-based** — recommends items similar in content (text embeddings + numeric metadata) to what a user already liked. Categories are included in the embedded text blob. Ignores other users; weak when item content is thin.
- **Collaborative filtering (CF)** — recommends from user behaviour patterns in the ratings matrix. Strong when data is dense; struggles with sparsity and cold-start.

The **hybrid** fuses both so each covers the other's weakness. The project is planned in staged modeling/infrastructure phases:

1. **Weighted hybrid** — `score = α · CF + (1 − α) · content`, implemented in Phase 1.
2. **Advanced content/review-feedback models** — planned next: filtered category features, review-text sentiment, user strictness/generosity features, and stronger sanity baselines.
3. **Graph recommenders** — planned after that: LightGCN and GraphSAGE over the user-item graph, with item/user features where appropriate.
4. **Retrieval/reasoning infrastructure** — Milvus + Neo4j + an LLM reasoning layer as the final extension, not the primary evaluated recommender.

## Datasets

Amazon Reviews 2023 categories are config-driven. The current dataset roles are:

| Category | Role | Notes |
|---|---|---|
| `Video_Games` | primary benchmark | true 5-core: 814,586 interactions, 94,762 users, 25,612 items |
| `Movies_and_TV` | second benchmark | true 5-core: 7,441,129 interactions, 657,203 users, 197,943 items |
| `Digital_Music` | sparsity/cold-start case study | strict 5-core empties it; even 2-core leaves only a tiny evaluable core |

Switch the active dataset in [`config/config.yaml`](config/config.yaml) — no code changes needed. Raw data is downloaded locally and is **not** committed.

Download a configured dataset with:

```bash
python -m src.data.fetch --dataset movies_and_tv
```

Then validate how much survives preprocessing:

```bash
python -m src.data.preprocess --dataset movies_and_tv
```

Processed artifacts are stored as Parquet plus an EDA JSON summary under `data/processed/<dataset>/`.
Raw and processed data are reproducible local artifacts and are not committed.

## Models

| Model | Signal used |
|---|---|
| Content-based | item content features |
| Item-KNN CF | ratings matrix |
| SVD CF (matrix factorization) | ratings matrix |
| Weighted hybrid | both (blended at output) |
| Enriched content/review models | item metadata + train-only review feedback (planned) |
| LightGCN / GraphSAGE | graph structure, optionally node features (planned) |

All models share one interface — `fit`, `predict(user, item)`, `recommend(user, K)` — so the evaluation harness and app treat them identically.

## Evaluation

Every model is evaluated with:

- **RMSE** / **MAE** — rating-prediction accuracy over held-out test rows
- **Precision@K** / **Recall@K** / **F1@K** — **sampled-candidate** ranking (relevant = rating ≥ 4): each user is ranked over its held-out positives + N sampled unseen negatives (N = 100, seed 42). These are sampled-candidate metrics, not full-catalog ranking.

Materialize/inspect item embeddings (optional — evaluation builds them on first run otherwise):

```bash
python -m src.models.embed --dataset <key> [--preview]
```

Run the model comparison (writes `data/processed/<key>/metrics.json`, a local artifact):

```bash
python -m src.evaluation.evaluate --dataset video_games --no-knn
python -m src.evaluation.evaluate --dataset movies_and_tv --no-knn --max-eval-users 5000
```

Advanced models (enriched content, sentiment, user/item aggregates, baselines, and α tuning):

[![Open sentiment feature builder in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/amazon-hybrid-recsys/blob/feat/advanced-models/notebooks/colab_sentiment_features.ipynb)

```bash
# (advanced) build train-only sentiment + user/item aggregates once per dataset
#   --fake = FakeSentimentModel (no HF download); omit it for the real HF model
./.venv/bin/python -m src.features.build_sentiment_features --dataset video_games --fake

# (advanced) add random/popularity baselines + enriched content + calibrated hybrid,
# and tune the hybrid alpha on a train-only validation slice
./.venv/bin/python -m src.evaluation.evaluate --dataset video_games --no-knn --advanced --tune-alpha
```

## First results

Full advanced run on the primary benchmark (`video_games`, 5-core, 79,248 ranking users).
The sentiment features were built from train-only review text with
`distilbert-base-uncased-finetuned-sst-2-english` on MPS:

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|
| content | 1.4757 | 1.0580 | 0.0399 | 0.2856 | 0.0675 |
| svd | **1.1337** | 0.8173 | 0.0335 | 0.2100 | 0.0543 |
| hybrid | **1.1337** | 0.8173 | 0.0335 | 0.2100 | 0.0543 |
| random | 1.2569 | 0.9780 | 0.0152 | 0.0975 | 0.0249 |
| popularity | 1.2270 | 0.9059 | **0.0776** | **0.5108** | **0.1284** |
| content_enriched | 1.2660 | 0.8258 | 0.0755 | 0.5014 | 0.1250 |
| calibrated_hybrid | 1.1654 | **0.7891** | 0.0528 | 0.3126 | 0.0848 |

SVD remains the strongest RMSE model, while calibrated hybrid gives the best MAE.
Popularity is a very strong sampled-ranking baseline; the enriched content model is
close behind it on P/R/F1 while improving substantially over plain content on RMSE/MAE.
The tuned alpha selected `1.0`, so the classic weighted hybrid collapses to SVD in this run.

Earlier sampled run on the second benchmark (`movies_and_tv`, 5,000 ranking users):

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|
| content | 1.2539 | 0.8699 | 0.0618 | 0.3588 | 0.0975 |
| svd | 1.0560 | 0.7503 | 0.0489 | 0.2521 | 0.0742 |
| hybrid | 1.0832 | 0.7890 | 0.0513 | 0.2743 | 0.0793 |

The P/R/F1 values are sampled-candidate metrics — compare them against the
random and popularity rows before judging their absolute scale.

`metrics.json` and embeddings under `data/processed/` are local, reproducible artifacts and are **not** committed.

## Roadmap

- **Phase 1** — data pipeline, content-based + KNN + SVD baselines, weighted hybrid, sampled-candidate evaluation.
- **Phase 2 (`feat/advanced-models`)** — richer content/review-feedback models: filtered categories, train-only review sentiment, user strictness/generosity features, popularity/random baselines, and hybrid calibration.
- **Phase 3 (`feat/graph-recommender`)** — LightGCN and GraphSAGE plus graph EDA/community analysis.
- **Phase 4 (`feat/streamlit-app`)** — visual app layer over metrics, EDA, users, items, and recommendations.
- **Phase 5 (`feat/storage-vector-graph-dbs`)** — Milvus vector search and Neo4j graph storage.
- **Phase 6 (`feat/llm-recommender-system`)** — LLM-guided explanation/reasoning over model outputs, vector search, and graph queries.

## Project structure

```
config/        central configuration (datasets, preprocessing, evaluation)
src/
  data/        download, parse, filter, split, build graph
  models/      content-based, KNN, SVD, weighted hybrid, advanced/graph models
  evaluation/  metrics and the comparison harness
  app/         Streamlit application
tests/         pytest suite
```

## Setup

Requires **Python 3.11**.

```bash
pip install -r requirements.txt
```

> Runs on **Python 3.11** (`scikit-surprise` / `sentence-transformers` need it). Deps are pinned in `requirements.txt`.

## Status

🚧 Phase 2 (`feat/advanced-models`): filtered category features, **train-only** review-text sentiment + user/item aggregates (consumed by the enriched content model), random + popularity baselines, and validation-slice α tuning. **Leakage rule:** held-out test review text never feeds the prediction for the same test interaction. Compare any low-looking P@10 against the **random**/**popularity** rows before judging a model.

- Models: content-based, SVD CF, Item-KNN CF, and a weighted hybrid behind one `fit/predict/recommend` interface, plus Granite/MiniLM embeddings (cached) and a sampled-negative evaluation runner.
- A first sampled `movies_and_tv` run is in (see [First results](#first-results)); `digital_music` is validated end-to-end (cold-start case study, not benchmark).
- **LightGCN/GraphSAGE**, Streamlit, Milvus/Neo4j, and the LLM recommender layer remain planned later phases.

Dataset roles: `Video_Games` and `Movies_and_TV` survive strict 5-core (the benchmarks); `Digital_Music` only survives at 2-core and is the sparsity/cold-start case study.
