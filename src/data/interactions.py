"""Clean the user-item interaction matrix: load, dedup, k-core filter, split."""

import numpy as np
import pandas as pd

INTERACTION_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]


def load_interactions(records):
    """Project review records to interaction rows, dropping invalid ones.

    Returns ``(df, raw_count)`` where ``raw_count`` is the number of records seen
    (before dropping) and ``df`` keeps only rows with user_id, parent_asin, rating.
    """
    raw_count = 0
    rows = []
    for r in records:
        raw_count += 1
        uid, iid, rating = r.get("user_id"), r.get("parent_asin"), r.get("rating")
        if uid is None or iid is None or rating is None:
            continue
        rows.append(
            {
                "user_id": uid,
                "parent_asin": iid,
                "rating": float(rating),
                "timestamp": r.get("timestamp"),
            }
        )
    df = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    return df, raw_count


def deduplicate_interactions(df):
    """Keep exactly one interaction per (user_id, parent_asin): the latest by timestamp.

    Stable sort makes the choice deterministic when timestamps are missing or tied.
    """
    ordered = df.sort_values("timestamp", kind="stable", na_position="first")
    deduped = ordered.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")
    return deduped.reset_index(drop=True)


def apply_k_core(df, k):
    """Iteratively keep users and items with >= k interactions until both hold."""
    current = df
    while True:
        user_counts = current["user_id"].value_counts()
        item_counts = current["parent_asin"].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        filtered = current[
            current["user_id"].isin(keep_users)
            & current["parent_asin"].isin(keep_items)
        ]
        if len(filtered) == len(current):
            return filtered.reset_index(drop=True)
        current = filtered


def split_per_user(df, test_size, random_seed, chronological=True):
    """Split each user's interactions into (train, test).

    When timestamps are present, the latest ``test_size`` fraction is held out
    (chronological holdout); otherwise a seeded random per-user split is used.
    Every user with >= 2 interactions gets at least one train and one test row.
    """
    rng = np.random.default_rng(random_seed)
    train_parts, test_parts = [], []
    for _, group in df.groupby("user_id", sort=True):
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue
        n_test = min(max(1, round(test_size * n)), n - 1)
        if chronological and group["timestamp"].notna().all():
            ordered = group.sort_values("timestamp", kind="stable")
        else:
            ordered = group.iloc[rng.permutation(n)]
        test_parts.append(ordered.iloc[-n_test:])
        train_parts.append(ordered.iloc[:-n_test])

    train = (
        pd.concat(train_parts).reset_index(drop=True)
        if train_parts
        else df.iloc[0:0].copy()
    )
    test = (
        pd.concat(test_parts).reset_index(drop=True)
        if test_parts
        else df.iloc[0:0].copy()
    )
    return train, test
