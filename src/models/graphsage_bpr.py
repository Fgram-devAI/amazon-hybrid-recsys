"""GraphSAGE-BPR recommender: feature-rich graph encoder with pairwise ranking loss.

This complements ``GraphSAGERecommender`` instead of replacing it:

- GraphSAGERecommender optimizes rating regression with MSE.
- GraphSAGEBPRRecommender optimizes top-K ranking with BPR, like LightGCN,
  but keeps the enriched user/item node features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.tune import split_validation
from src.graph.build import BipartiteGraph
from src.models.graphsage import GraphSAGERecommender, _SAGEEdgeModel


class GraphSAGEBPRRecommender(GraphSAGERecommender):
    def __init__(
        self,
        *,
        embedder: object,
        generic_roots: list[str],
        max_vocab: int,
        min_doc_freq: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        epochs: int = 20,
        lr: float = 0.005,
        num_negatives: int = 1,
        batch_size: int = 1024,
        seed: int = 42,
        device: str = "auto",
        cache_dir: object = None,
        review_features_dir: object = None,
        validation_fraction: float = 0.1,
        progress: bool = False,
    ) -> None:
        super().__init__(
            embedder=embedder,
            generic_roots=generic_roots,
            max_vocab=max_vocab,
            min_doc_freq=min_doc_freq,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            device=device,
            cache_dir=cache_dir,
            review_features_dir=review_features_dir,
            progress=progress,
        )
        self.num_negatives = int(num_negatives)
        self.validation_fraction = float(validation_fraction)
        self._calibration_beta: float | None = None
        self._calibration_intercept: float | None = None

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "GraphSAGEBPRRecommender":
        train_only, val = split_validation(
            train, validation_fraction=self.validation_fraction, seed=self.seed,
        )
        if val.empty:
            train_only, val = train, train.head(min(64, len(train)))

        if self.progress:
            print(
                f"[graphsage_bpr] calibration split: train_only={len(train_only):,}, "
                f"val={len(val):,}",
                flush=True,
            )
        cal_edge_index, _, _ = self._prepare_state(train_only, metadata)
        assert self._model is not None
        assert self._x is not None
        cal_model = self._train_bpr(self._graph, cal_edge_index, self._model)
        self._calibration_beta, self._calibration_intercept = self._fit_calibration(
            cal_model, self._graph, val, cal_edge_index,
        )
        if self.progress:
            print(
                f"[graphsage_bpr] calibration head: beta={self._calibration_beta:.4f}, "
                f"intercept={self._calibration_intercept:.4f}",
                flush=True,
            )

        edge_index, _, _ = self._prepare_state(train, metadata)
        assert self._model is not None
        assert self._x is not None
        self._model = self._train_bpr(self._graph, edge_index, self._model)
        self.cache_final_embeddings(edge_index)
        return self

    def _train_bpr(
        self,
        graph: BipartiteGraph | None,
        edge_index: torch.Tensor,
        model: _SAGEEdgeModel,
    ) -> _SAGEEdgeModel:
        if graph is None:
            raise ValueError("GraphSAGE-BPR graph state is not prepared")
        assert self._x is not None
        pos_edges = graph.positive_edge_label_index.to(self.device)
        if pos_edges.shape[1] == 0:
            return model.eval()

        optimizer = torch.optim.Adam(model.convs.parameters(), lr=self.lr)
        n_items = len(graph.item_index)
        item_offset = graph.item_offset
        rng = np.random.default_rng(self.seed)
        positives_by_user = self._positive_items_by_user(graph)

        model.train()
        for epoch in range(self.epochs):
            h = model.encode(self._x, edge_index)
            perm = torch.randperm(pos_edges.shape[1], device=self.device)
            optimizer.zero_grad()
            n_batches = int(np.ceil(perm.shape[0] / self.batch_size))
            epoch_loss = 0.0
            batch_starts = range(0, perm.shape[0], self.batch_size)
            for batch_idx, start in enumerate(tqdm(
                batch_starts,
                total=n_batches,
                desc=f"[graphsage_bpr] epoch {epoch + 1}/{self.epochs}",
                unit="batch",
                disable=not self.progress,
                leave=False,
            )):
                idx = perm[start:start + self.batch_size]
                users = pos_edges[0, idx]
                pos_items = pos_edges[1, idx]
                neg_items = self._sample_negatives(
                    users.cpu().numpy(),
                    n_items=n_items,
                    item_offset=item_offset,
                    positives_by_user=positives_by_user,
                    rng=rng,
                ).to(self.device)
                user_emb = h[users]
                pos_emb = h[pos_items]
                neg_emb = h[neg_items]
                pos_score = (user_emb * pos_emb).sum(dim=-1, keepdim=True)
                neg_score = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
                loss = F.softplus(neg_score - pos_score).mean()
                epoch_loss += float(loss.detach().cpu())
                loss.backward(retain_graph=batch_idx != n_batches - 1)
            optimizer.step()
            if self.progress:
                print(
                    f"[graphsage_bpr] epoch {epoch + 1}/{self.epochs}: "
                    f"batches={n_batches:,}, mean_bpr_loss={epoch_loss / max(n_batches, 1):.4f}",
                    flush=True,
                )
        return model.eval()

    def _positive_items_by_user(self, graph: BipartiteGraph) -> dict[int, set[int]]:
        positives: dict[int, set[int]] = {}
        edges = graph.positive_edge_label_index.cpu().numpy()
        for user, item_global in zip(edges[0], edges[1]):
            positives.setdefault(int(user), set()).add(int(item_global - graph.item_offset))
        return positives

    def _sample_negatives(
        self,
        users: np.ndarray,
        *,
        n_items: int,
        item_offset: int,
        positives_by_user: dict[int, set[int]],
        rng: np.random.Generator,
    ) -> torch.Tensor:
        negatives = np.empty((len(users), self.num_negatives), dtype=np.int64)
        for row, user in enumerate(users):
            seen = positives_by_user.get(int(user), set())
            if len(seen) >= n_items:
                negatives[row, :] = rng.integers(0, n_items, size=self.num_negatives)
                continue
            for col in range(self.num_negatives):
                candidate = int(rng.integers(0, n_items))
                attempts = 0
                while candidate in seen and attempts < 100:
                    candidate = int(rng.integers(0, n_items))
                    attempts += 1
                if candidate in seen:
                    for fallback in range(n_items):
                        if fallback not in seen:
                            candidate = fallback
                            break
                negatives[row, col] = candidate
        return torch.from_numpy(negatives + item_offset)

    def _fit_calibration(
        self,
        model: _SAGEEdgeModel,
        graph: BipartiteGraph | None,
        val: pd.DataFrame,
        edge_index: torch.Tensor,
    ) -> tuple[float, float]:
        if graph is None or val.empty:
            return 0.0, float(self.global_mean_)
        assert self._x is not None
        u_local: list[int] = []
        i_global: list[int] = []
        ratings: list[float] = []
        for _, row in val.iterrows():
            uid = graph.user_index.get(str(row["user_id"]))
            iid = graph.item_index.get(str(row["parent_asin"]))
            if uid is None or iid is None:
                continue
            u_local.append(uid)
            i_global.append(iid + graph.item_offset)
            ratings.append(float(row["rating"]))
        if not u_local:
            return 0.0, float(self.global_mean_)
        u_t = torch.tensor(u_local, dtype=torch.long, device=self.device)
        i_t = torch.tensor(i_global, dtype=torch.long, device=self.device)
        with torch.no_grad():
            h = model.encode(self._x, edge_index)
            raw = (h[u_t] * h[i_t]).sum(dim=-1).cpu().numpy()
        y = np.asarray(ratings, dtype=np.float32)
        if raw.std() == 0:
            return 0.0, float(y.mean())
        beta, intercept = np.polyfit(raw, y, 1)
        return float(beta), float(intercept)

    def predict(self, user_id: str, parent_asin: str) -> float:
        if self._model is None or self._graph is None or self._final_embeddings is None:
            return self._clip(self._fallback(user_id, parent_asin))
        uid = self._graph.user_index.get(user_id)
        iid = self._graph.item_index.get(parent_asin)
        if uid is None or iid is None:
            return self._clip(self._fallback(user_id, parent_asin))
        h = self._final_embeddings
        raw = float((h[uid] * h[iid + self._graph.item_offset]).sum().item())
        beta = self._calibration_beta if self._calibration_beta is not None else 0.0
        intercept = (
            self._calibration_intercept
            if self._calibration_intercept is not None
            else self.global_mean_
        )
        return self._clip(beta * raw + intercept)

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["calibration_beta"] = self._calibration_beta
        state["calibration_intercept"] = self._calibration_intercept
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self._calibration_beta = state.get("calibration_beta")  # type: ignore[assignment]
        self._calibration_intercept = state.get("calibration_intercept")  # type: ignore[assignment]
