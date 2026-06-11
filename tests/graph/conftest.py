"""Shared synthetic-graph fixtures for tests/graph/."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def coratings_toy() -> pd.DataFrame:
    """5 users x 3 items co-rating pattern with hand-computable shares.

    u1, u2, u3, u4 all rate i1 and i2  -> shared_users(i1, i2) = 4
    u1 also rates i3                   -> shared_users(i1, i3) = 1
    u5 rates only i3                   -> i3 union with i1 is {u1,u2,u3,u4,u5}
    """
    rows = [
        ("u1", "i1"), ("u1", "i2"), ("u1", "i3"),
        ("u2", "i1"), ("u2", "i2"),
        ("u3", "i1"), ("u3", "i2"),
        ("u4", "i1"), ("u4", "i2"),
        ("u5", "i3"),
    ]
    return pd.DataFrame(
        [
            {"user_id": u, "parent_asin": it, "rating": 5.0, "timestamp": idx}
            for idx, (u, it) in enumerate(rows)
        ]
    )


@pytest.fixture
def two_clique_train() -> pd.DataFrame:
    """Two planted item communities with a single weak bridge.

    Cluster A = {iA1, iA2, iA3}, all co-rated by uA1..uA4.
    Cluster B = {iB1, iB2, iB3}, all co-rated by uB1..uB4.
    Bridge: uX rates iA1 and iB1 -> single weak edge between clusters.
    """
    rows: list[tuple[str, str]] = []
    for u in ("uA1", "uA2", "uA3", "uA4"):
        for it in ("iA1", "iA2", "iA3"):
            rows.append((u, it))
    for u in ("uB1", "uB2", "uB3", "uB4"):
        for it in ("iB1", "iB2", "iB3"):
            rows.append((u, it))
    rows.append(("uX", "iA1"))
    rows.append(("uX", "iB1"))
    return pd.DataFrame(
        [
            {"user_id": u, "parent_asin": it, "rating": 5.0, "timestamp": idx}
            for idx, (u, it) in enumerate(rows)
        ]
    )
