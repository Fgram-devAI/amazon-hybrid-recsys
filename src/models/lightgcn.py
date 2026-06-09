"""LightGCN recommender wrapping torch_geometric.nn.models.LightGCN.

Native objective: BPR for top-K ranking. predict() returns a calibrated rating
computed by least-squares-fitting (beta, intercept) on a train-derived validation
slice (NEVER on test). Calibrated rating is clipped to [1, 5].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN

from src.evaluation.tune import split_validation
from src.graph.build import BipartiteGraph, build_graph
from src.models.graph_base import GraphRecommender


class LightGCNRecommender(GraphRecommender):
    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        n_layers: int = 3,
        epochs: int = 20,
        lr: float = 0.001,
        num_negatives: int = 1,
        batch_size: int = 1024,
        seed: int = 42,
        device: str = "auto",
        min_rating_positive: float = 4.0,
        validation_fraction: float = 0.1,
    ) -> None:
        super().__init__(device=device)
        self.embedding_dim = int(embedding_dim)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.num_negatives = int(num_negatives)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_rating_positive = float(min_rating_positive)
        self.validation_fraction = float(validation_fraction)
        self._graph: BipartiteGraph | None = None
        self._model: LightGCN | None = None
        self._calibration_beta: float | None = None
        self._calibration_intercept: float | None = None
        self._final_embeddings: torch.Tensor | None = None

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "LightGCNRecommender":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_means(train)

        # train -> (train_only, val) for calibration
        train_only, val = split_validation(
            train, validation_fraction=self.validation_fraction, seed=self.seed,
        )
        if val.empty:
            train_only, val = train, train.head(min(64, len(train)))

        # Build a graph over train_only first to calibrate, then refit on full train
        cal_graph = build_graph(train_only, min_rating_positive=self.min_rating_positive)
        cal_model = self._train_bpr(cal_graph)
        self._calibration_beta, self._calibration_intercept = self._fit_calibration(
            cal_model, cal_graph, val,
        )

        # Final fit on full train
        self._graph = build_graph(train, min_rating_positive=self.min_rating_positive)
        self._model = self._train_bpr(self._graph)

        # Cache final propagated embeddings so predict() is O(D) per call, not
        # O(full-graph forward) — eval loops do tens of thousands of predicts.
        final_prop = self._graph.positive_propagation_edge_index.to(self.device)
        with torch.no_grad():
            self._final_embeddings = self._model.get_embedding(final_prop).detach()
        return self

    def _train_bpr(self, graph: BipartiteGraph) -> LightGCN:
        model = LightGCN(
            num_nodes=graph.num_nodes,
            embedding_dim=self.embedding_dim,
            num_layers=self.n_layers,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        pos_edges = graph.positive_edge_label_index.to(self.device)
        if pos_edges.shape[1] == 0:
            return model.eval()
        prop = graph.positive_propagation_edge_index.to(self.device)
        n_items = len(graph.item_index)
        item_offset = graph.item_offset
        rng = np.random.default_rng(self.seed)
        positives_by_user = self._positive_items_by_user(graph)

        model.train()
        for _ in range(self.epochs):
            h = model.get_embedding(prop)
            perm = torch.randperm(pos_edges.shape[1], device=self.device)
            optimizer.zero_grad()
            n_batches = int(np.ceil(perm.shape[0] / self.batch_size))
            for start in range(0, perm.shape[0], self.batch_size):
                idx = perm[start:start + self.batch_size]
                u = pos_edges[0, idx]
                pos_i = pos_edges[1, idx]
                neg_i = self._sample_negatives(
                    u.cpu().numpy(),
                    n_items=n_items,
                    item_offset=item_offset,
                    positives_by_user=positives_by_user,
                    rng=rng,
                ).to(self.device)
                u_emb = h[u]
                pos_emb = h[pos_i]
                neg_emb = h[neg_i]
                pos_score = (u_emb.unsqueeze(1) * pos_emb.unsqueeze(1)).sum(dim=-1)
                neg_score = (u_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
                loss = F.softplus(neg_score - pos_score).mean()
                is_last = (start // self.batch_size) == n_batches - 1
                loss.backward(retain_graph=not is_last)
            optimizer.step()
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
        """Sample item negatives not present in each user's positive train set."""
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
        model: LightGCN,
        graph: BipartiteGraph,
        val: pd.DataFrame,
    ) -> tuple[float, float]:
        if val.empty:
            return 0.0, float(self.global_mean_)
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
        prop = graph.positive_propagation_edge_index.to(self.device)
        with torch.no_grad():
            h = model.get_embedding(prop)
            raw = (h[u_t] * h[i_t]).sum(dim=-1).cpu().numpy()
        y = np.asarray(ratings, dtype=np.float32)
        if raw.std() == 0:
            return 0.0, float(y.mean())
        # simple linear: rating ~ beta * raw + intercept
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
        assert self._model is not None
        return {
            "model": self._model.state_dict(),
            "calibration_beta": self._calibration_beta,
            "calibration_intercept": self._calibration_intercept,
            "final_embeddings": self._final_embeddings,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert self._model is not None
        self._model.load_state_dict(state["model"])  # type: ignore[arg-type]
        self._calibration_beta = state.get("calibration_beta")  # type: ignore[assignment]
        self._calibration_intercept = state.get("calibration_intercept")  # type: ignore[assignment]
        self._final_embeddings = state.get("final_embeddings")  # type: ignore[assignment]
