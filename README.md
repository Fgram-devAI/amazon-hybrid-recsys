# Amazon Hybrid Recommender System

A hybrid recommender system that combines **content-based filtering** and **collaborative filtering**, built on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. It compares the hybrid approach against standalone baselines on standard metrics and serves recommendations through a Streamlit app.

> MSc Artificial Intelligence — *Applications in AI*, Topic 2 (Hybrid Recommender System).

## Overview

Two recommendation paradigms, each with a known weakness:

- **Content-based** — recommends items similar in content (text embeddings + genre/brand/price) to what a user already liked. Ignores other users; weak when item content is thin.
- **Collaborative filtering (CF)** — recommends from user behaviour patterns in the ratings matrix. Strong when data is dense; struggles with sparsity and cold-start.

The **hybrid** fuses both so each covers the other's weakness. This project implements the fusion **two ways**:

1. **Weighted hybrid** — `score = α · CF + (1 − α) · content`, with `α` tuned per dataset.
2. **GraphSAGE hybrid** — a Graph Neural Network over the user–item graph, where item nodes carry content features and message passing captures collaborative structure. The two signals are fused *inside the model*, not blended afterward.

## Datasets

Two categories from Amazon Reviews 2023, chosen for contrast:

| Category | Density | Highlights |
|---|---|---|
| `Video_Games` | denser | collaborative filtering strengths |
| `Digital_Music` | sparse | content-based / cold-start strengths |

Switch the active dataset in [`config/config.yaml`](config/config.yaml) — no code changes needed. Raw data is downloaded locally and is **not** committed.

## Models

| Model | Signal used |
|---|---|
| Content-based | item content features |
| Item-KNN CF | ratings matrix |
| SVD CF (matrix factorization) | ratings matrix |
| Weighted hybrid | both (blended at output) |
| GraphSAGE hybrid | both (fused in-model) |

All models share one interface — `fit`, `predict(user, item)`, `recommend(user, K)` — so the evaluation harness and app treat them identically.

## Evaluation

Every model is evaluated on both datasets with:

- **RMSE** / **MAE** — rating-prediction accuracy
- **Precision@K** / **Recall@K** / **F1@K** — top-K ranking quality (relevant = rating ≥ 4)

## Roadmap

- **Phase 1** — data pipeline, content-based + KNN + SVD baselines, weighted hybrid, full evaluation, Streamlit app.
- **Phase 2** — GraphSAGE hybrid (PyTorch Geometric) added as a fifth model.
- **Phase 3** — Neo4j graph store + LightRAG (Claude) for explainable, conversational recommendations.

## Project structure

```
config/        central configuration (datasets, preprocessing, evaluation)
src/
  data/        download, parse, filter, split, build graph
  models/      content-based, KNN, SVD, weighted hybrid, GraphSAGE
  evaluation/  metrics and the comparison harness
  app/         Streamlit application
tests/         pytest suite
```

## Setup

Requires **Python 3.11**.

```bash
pip install -r requirements.txt
```

> Dependency management is being migrated to `uv` + `pyproject.toml`, with optional groups per phase (`gnn`, `graph`).

## Status

🚧 In active development — Phase 1.
