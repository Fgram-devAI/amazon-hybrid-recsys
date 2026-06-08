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
| GraphSAGE hybrid | both (fused in-model) |

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

## First results

Sampled run on the second benchmark (`movies_and_tv`, 5,000 ranking users):

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|
| content | 1.2539 | 0.8699 | 0.0618 | 0.3588 | 0.0975 |
| svd | 1.0560 | 0.7503 | 0.0489 | 0.2521 | 0.0742 |
| hybrid | 1.0832 | 0.7890 | 0.0513 | 0.2743 | 0.0793 |

SVD currently wins rating prediction; content wins sampled ranking. The hybrid does not yet beat both because `alpha` is fixed at 0.5 (not yet tuned/calibrated). The P/R/F1 are sampled-candidate metrics — a random baseline on the same setup is ≈ P@10 0.019 / R@10 0.098 / F1@10 0.029, so content is well above random even though precision looks numerically low.

`metrics.json` and embeddings under `data/processed/` are local, reproducible artifacts and are **not** committed.

## Roadmap

- **Phase 1** — data pipeline, content-based + KNN + SVD baselines, weighted hybrid, full evaluation, Streamlit app.
- **Phase 2** — GraphSAGE hybrid (PyTorch Geometric) added as a fifth model in the main implementation.
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

> Runs on **Python 3.11** (`scikit-surprise` / `sentence-transformers` need it). Deps are pinned in `requirements.txt`.

## Status

🚧 Phase 1 — **`feat/models` implemented; evaluation underway.**

- Models: content-based, SVD CF, Item-KNN CF, and a weighted hybrid behind one `fit/predict/recommend` interface, plus Granite/MiniLM embeddings (cached) and a sampled-negative evaluation runner.
- A first sampled `movies_and_tv` run is in (see [First results](#first-results)); `digital_music` is validated end-to-end (cold-start case study, not benchmark).
- **GraphSAGE** remains planned for **Phase 2** (not implemented).

Dataset roles: `Video_Games` and `Movies_and_TV` survive strict 5-core (the benchmarks); `Digital_Music` only survives at 2-core and is the sparsity/cold-start case study.
