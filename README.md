# Amazon Hybrid Recommender System

A hybrid recommender system that combines **content-based filtering** and **collaborative filtering**, built on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. It compares the hybrid approach against standalone baselines on standard metrics and serves recommendations through a Streamlit app.

> MSc Artificial Intelligence — *Applications in AI*, Topic 2 (Hybrid Recommender System).

## Overview

Two recommendation paradigms, each with a known weakness:

- **Content-based** — recommends items similar in content (text embeddings + numeric metadata) to what a user already liked. Categories are included in the embedded text blob. Ignores other users; weak when item content is thin.
- **Collaborative filtering (CF)** — recommends from user behaviour patterns in the ratings matrix. Strong when data is dense; struggles with sparsity and cold-start.

The **hybrid** fuses both so each covers the other's weakness. The project is planned in staged modeling/infrastructure phases:

1. **Weighted hybrid** — `score = α · CF + (1 − α) · content`, implemented in Phase 1.
2. **Advanced content/review-feedback models** — filtered category features, review-text sentiment, user strictness/generosity features, and stronger sanity baselines.
3. **Graph recommenders** — LightGCN and GraphSAGE over the user-item graph, with item/user features where appropriate.
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
| Enriched content/review models | item metadata + train-only review feedback |
| LightGCN / GraphSAGE / GraphSAGE-BPR | train-only user-item graph, optionally node features |

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
`distilbert-base-uncased-finetuned-sst-2-english` on MPS. The table below uses
the RMSE-tuned alpha (`α = 1.0`):

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|
| content | 1.4757 | 1.0580 | 0.0399 | 0.2856 | 0.0675 |
| svd | **1.1337** | 0.8173 | 0.0335 | 0.2100 | 0.0543 |
| hybrid | **1.1337** | 0.8173 | 0.0335 | 0.2100 | 0.0543 |
| random | 1.2569 | 0.9780 | 0.0152 | 0.0975 | 0.0249 |
| popularity | 1.2270 | 0.9059 | **0.0776** | **0.5108** | **0.1284** |
| content_enriched | 1.2660 | 0.8258 | 0.0755 | 0.5014 | 0.1250 |
| calibrated_hybrid | 1.1654 | 0.7891 | 0.0528 | 0.3126 | 0.0848 |

SVD remains the strongest RMSE model. Popularity is a very strong sampled-ranking
baseline; the enriched content model is close behind it on P/R/F1 while improving
substantially over plain content on RMSE/MAE. The RMSE-tuned alpha selected `1.0`,
so the classic weighted hybrid collapses to SVD in this run.

Fixed-alpha sweep for the calibrated hybrid (`α · SVD + (1 - α) · content_enriched`):

| α | RMSE | MAE | P@10 | R@10 | F1@10 |
|---:|---:|---:|---:|---:|---:|
| 0.75 | **1.1501** | 0.7698 | 0.0556 | 0.3330 | 0.0896 |
| 0.60 | 1.1596 | **0.7664** | 0.0588 | 0.3562 | 0.0950 |
| 0.50 | 1.1737 | 0.7679 | 0.0619 | 0.3789 | 0.1003 |
| 0.40 | 1.1938 | 0.7722 | **0.0662** | **0.4093** | **0.1076** |

This sweep shows the expected trade-off: lower `α` uses more enriched-content signal
and improves hybrid ranking, while higher `α` stays closer to SVD for rating RMSE.
The standalone `content_enriched` row remains the strongest advanced-content ranking
result (`F1@10 = 0.1250`), narrowly below popularity (`F1@10 = 0.1284`).

### Sentiment ablation (`video_games`, sampled-candidate, α = 0.6)

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|
| content_enriched_with_sentiment | 1.2660 | 0.8258 | 0.0755 | 0.5014 | 0.1250 |
| content_enriched_no_sentiment   | 1.1777 | 0.7917 | 0.0627 | 0.4278 | 0.1047 |

The no-sentiment variant uses the same train slice, embedder, sampled
candidates, and category vocabulary as the sentiment-aware variant; it
drops only the train-only item-sentiment columns and the user-generosity
offset.

Sentiment improves F1@10 from 0.1047 to 0.1250 (Δ = +0.0204) but regresses MAE from 0.7917 to 0.8258 (Δ = +0.0341), so sentiment is kept as an optional ranking-oriented feature rather than a universal win.

Reproduction command:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games --no-knn --advanced --include-ablation --alpha 0.6
```

Earlier sampled run on the second benchmark (`movies_and_tv`, 5,000 ranking users):

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|
| content | 1.2539 | 0.8699 | 0.0618 | 0.3588 | 0.0975 |
| svd | 1.0560 | 0.7503 | 0.0489 | 0.2521 | 0.0742 |
| hybrid | 1.0832 | 0.7890 | 0.0513 | 0.2743 | 0.0793 |

The P/R/F1 values are sampled-candidate metrics — compare them against the
random and popularity rows before judging their absolute scale.

### Graph checkpoint eval (`video_games`)

Checkpoint-based graph evaluation on the primary benchmark using the full held-out
split (`79,248` ranking users).

| Model | RMSE | MAE | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|
| LightGCN 10ep / neg1 | 1.2472 | **0.9556** | 0.0880 | 0.5755 | 0.1454 |
| LightGCN 20ep / neg1 | 1.2459 | 0.9589 | 0.0882 | 0.5782 | 0.1458 |
| LightGCN 40ep / neg4 | 1.2433 | 0.9598 | 0.0901 | 0.5923 | 0.1491 |
| LightGCN 40ep / neg4 / wd1e-5 | 1.2433 | 0.9598 | **0.0902** | **0.5928** | **0.1492** |
| GraphSAGE MSE 10ep | **1.1613** | **0.7640** | 0.0235 | 0.1488 | 0.0380 |
| GraphSAGE-BPR 20ep / neg4 | 1.2559 | 0.9814 | 0.0550 | 0.3573 | 0.0906 |

LightGCN is the stronger ranking model, which matches its BPR top-K objective.
Increasing epochs alone (`10 -> 20`) was almost flat, but increasing BPR negative
sampling (`40ep / neg4`) gave a modest, consistent ranking lift; light weight decay
nudged it slightly further. GraphSAGE with MSE remains the better graph rating
predictor, while GraphSAGE-BPR substantially improves GraphSAGE ranking but still
trails LightGCN.

`metrics.json` and embeddings under `data/processed/` are local, reproducible artifacts and are **not** committed.

## Graph Recommender Models (LightGCN + GraphSAGE)

The graph models extend the comparison table with PyTorch Geometric–backed
recommenders fitted on the train-only bipartite user-item graph (test interactions
are never part of the message-passing graph):

- **LightGCN** — pure-collaborative, BPR objective for top-K ranking. Its RMSE/MAE
  row comes from a calibrated score-to-rating head whose `(beta, intercept)` are
  fit on a validation slice carved from train, so the comparison row stays honest.
- **GraphSAGE** — content + collaborative fusion. Item nodes carry the enriched
  feature matrix (text embedding ⊕ filtered categories ⊕ numeric ⊕ train-only
  item-sentiment); user nodes carry train-only behavioural aggregates. Trained
  as edge regression on observed train ratings (the rating is never an input feature).
- **GraphSAGE-BPR** — same enriched GraphSAGE encoder, but trained with pairwise
  BPR on `rating >= 4` positives. This tests whether content/user node features
  can help graph ranking when the objective matches top-K recommendation.

Run graph training/evaluation:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games --graph-only --max-eval-users 5000
```

Train one graph model without evaluation, preserving the baseline checkpoint:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games \
  --graph-only \
  --only-model lightgcn \
  --graph-epochs 20 \
  --checkpoint-tag 20ep \
  --train-only
```

This writes `data/processed/video_games/graph_checkpoints/lightgcn_20ep.pt`
without overwriting the 10-epoch `lightgcn.pt` baseline.

Current LightGCN evidence: moving from 10 to 20 epochs was almost flat on the
full Video_Games checkpoint eval (`F1@10 0.1454 -> 0.1458`). The next useful
ranking experiment should change BPR negative sampling, not just epochs:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games \
  --graph-only \
  --only-model lightgcn \
  --graph-epochs 40 \
  --graph-num-negatives 4 \
  --checkpoint-tag 40ep_neg4 \
  --train-only
```

Then evaluate the tagged checkpoint:

```bash
./.venv/bin/python -m src.evaluation.evaluate_lightgcn_checkpoint \
  --dataset video_games \
  --checkpoint data/processed/video_games/graph_checkpoints/lightgcn_40ep_neg4.pt \
  --output data/processed/video_games/metrics_lightgcn_40ep_neg4_full.json
```

`40ep_neg4` improved held-out F1@10 modestly (`0.1454 -> 0.1491`), and light
L2 regularization nudged it to `0.1492`. If a later longer run flattens or hurts
ranking, prefer regularization/negative-sampling checks before adding more epochs:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games \
  --graph-only \
  --only-model lightgcn \
  --graph-epochs 40 \
  --graph-num-negatives 4 \
  --graph-weight-decay 1e-5 \
  --checkpoint-tag 40ep_neg4_wd1e-5 \
  --train-only
```

Train GraphSAGE-BPR without overwriting regression GraphSAGE:

```bash
./.venv/bin/python -m src.evaluation.evaluate \
  --dataset video_games \
  --graph-only \
  --only-model graphsage_bpr \
  --graph-epochs 20 \
  --graph-num-negatives 4 \
  --checkpoint-tag 20ep_neg4 \
  --train-only
```

Then evaluate the tagged checkpoint:

```bash
./.venv/bin/python -m src.evaluation.evaluate_graphsage_bpr_checkpoint \
  --dataset video_games \
  --checkpoint data/processed/video_games/graph_checkpoints/graphsage_bpr_20ep_neg4.pt \
  --output data/processed/video_games/metrics_graphsage_bpr_20ep_neg4_full.json
```

Re-evaluate stored graph checkpoints without retraining:

```bash
./.venv/bin/python -m src.evaluation.evaluate_lightgcn_checkpoint \
  --dataset video_games --max-eval-users 5000 --max-test-rows 50000

./.venv/bin/python -m src.evaluation.evaluate_graphsage_checkpoint \
  --dataset video_games --max-eval-users 5000 --max-test-rows 50000
```

When comparing longer graph training runs, save checkpoints under distinct names
such as `lightgcn_10ep.pt` and `lightgcn_20ep.pt` so the 10-epoch baseline remains
re-evaluable.

GraphSAGE is currently a rating-regression graph model: its 10-epoch checkpoint
wins RMSE/MAE against LightGCN but is weak on sampled ranking. Do not expect
ranking gains from simply increasing epochs; a future ranking-oriented GraphSAGE
variant should change the objective/head rather than only `lr` or `epochs`.
GraphSAGE-BPR is that ranking-oriented variant.

Graph EDA/community detection (item-item projections, Louvain/Leiden/spectral
clustering, category alignment) should live on a separate branch/spec because it
is interpretability/analysis work, not model training.

Dependencies installed once via `pip install -r requirements.txt` (heavy: torch
pulls ~2 GB). PyG 2.6 needs no separate `torch-scatter` / `torch-sparse`. Graph
device selection follows `cuda -> mps -> cpu`; MPS is opportunistic on Apple
Silicon because PyG operator coverage can vary.

## Roadmap

- **Phase 1** — data pipeline, content-based + KNN + SVD baselines, weighted hybrid, sampled-candidate evaluation.
- **Phase 2 (`feat/advanced-models`)** — richer content/review-feedback models: filtered categories, train-only review sentiment, user strictness/generosity features, popularity/random baselines, and hybrid calibration.
- **Phase 3 (`feat/graph-recommender`)** — LightGCN and GraphSAGE graph recommenders; graph EDA/community analysis follows separately.
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

Local generated outputs live under `data/processed/<dataset>/`:

```
train.parquet / test.parquet / metadata.parquet
eda_summary.json
embeddings/
advanced_features/
graph_checkpoints/
  lightgcn.pt
  graphsage.pt
  graphsage_bpr.pt
  lightgcn_20ep.pt        # optional tagged rerun
  lightgcn_40ep_neg4.pt   # optional BPR negative-sampling rerun
  lightgcn_40ep_neg4_wd1e-5.pt
metrics.json              # latest normal evaluator run
metrics_lightgcn_checkpoint.json
metrics_graphsage_checkpoint.json
metrics_graphsage_bpr_checkpoint.json
```

## Setup

Requires **Python 3.11**.

```bash
pip install -r requirements.txt
```

> Runs on **Python 3.11** (`scikit-surprise` / `sentence-transformers` need it). Deps are pinned in `requirements.txt`.

## Status

🚧 Phase 3 (`feat/graph-recommender`): LightGCN and GraphSAGE are implemented over the train-only bipartite graph, with checkpoint evaluators for long graph runs. **Leakage rule:** held-out test edges never enter message passing or node-feature aggregates. Compare any low-looking P@10 against the **random**/**popularity** rows before judging a model.

- Models: content-based, SVD CF, Item-KNN CF, and a weighted hybrid behind one `fit/predict/recommend` interface, plus Granite/MiniLM embeddings (cached) and a sampled-negative evaluation runner.
- A first sampled `movies_and_tv` run and capped graph checkpoint eval are in (see [First results](#first-results)); `digital_music` is validated end-to-end (cold-start case study, not benchmark).
- Streamlit, Milvus/Neo4j, and the LLM recommender layer remain planned later phases.

Dataset roles: `Video_Games` and `Movies_and_TV` survive strict 5-core (the benchmarks); `Digital_Music` only survives at 2-core and is the sparsity/cold-start case study.
