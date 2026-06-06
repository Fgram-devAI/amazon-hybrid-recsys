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
