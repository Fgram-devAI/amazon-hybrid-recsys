"""Tests for the EDA summary."""

import pandas as pd

from src.data.eda import summarize_eda


def _interactions(pairs_ratings):
    return pd.DataFrame(
        [
            {"user_id": u, "parent_asin": i, "rating": r, "timestamp": t}
            for t, (u, i, r) in enumerate(pairs_ratings)
        ]
    )


def test_summarize_eda_reports_counts_sparsity_and_relevance():
    # deduped is larger than kcore so the _before vs _after keys differ and can
    # each be asserted independently (kcore drops the sparse u3 / i3 rows).
    deduped = _interactions(
        [
            ("u1", "i1", 5.0), ("u1", "i2", 4.0),
            ("u2", "i1", 3.0), ("u2", "i2", 5.0),
            ("u3", "i3", 2.0),
        ]
    )
    kcore = _interactions(
        [("u1", "i1", 5.0), ("u1", "i2", 4.0), ("u2", "i1", 3.0), ("u2", "i2", 5.0)]
    )
    train = kcore.iloc[:2]
    test = kcore.iloc[2:]

    summary = summarize_eda(
        raw_count=100,
        valid_count=80,
        deduped_df=deduped,
        kcore_df=kcore,
        train_df=train,
        test_df=test,
        min_rating_relevant=4.0,
    )

    assert summary["raw_reviews"] == 100
    assert summary["after_drop_invalid"] == 80
    # before/after differ -> the k-core effect is actually captured
    assert summary["after_dedup"] == 5
    assert summary["after_kcore"] == 4
    assert summary["users_before"] == 3 and summary["users_after"] == 2
    assert summary["items_before"] == 3 and summary["items_after"] == 2
    assert summary["min_user_interactions_after"] == 2
    assert summary["min_item_interactions_after"] == 2
    assert summary["train_interactions"] == 2
    assert summary["test_interactions"] == 2
    # 3 of 4 ratings are >= 4.0
    assert summary["pct_relevant_after"] == 0.75
    # 2 users x 2 items, 4 cells filled -> sparsity 0.0
    assert summary["sparsity_after"] == 0.0
    assert summary["sparsity_before"] != summary["sparsity_after"]
    assert summary["rating_hist_after"]["5.0"] == 2


def test_summarize_eda_on_empty_kcore_is_json_serializable():
    import json

    empty = pd.DataFrame(columns=["user_id", "parent_asin", "rating", "timestamp"])
    deduped = _interactions([("u1", "i1", 5.0)])

    summary = summarize_eda(
        raw_count=10,
        valid_count=9,
        deduped_df=deduped,
        kcore_df=empty,
        train_df=empty,
        test_df=empty,
        min_rating_relevant=4.0,
    )

    # no NaN -> valid JSON (an over-aggressive k_core can empty a sparse set)
    assert summary["pct_relevant_after"] is None
    assert summary["min_user_interactions_after"] == 0
    assert summary["min_item_interactions_after"] == 0
    assert summary["rating_hist_before"]["5.0"] == 1
    assert summary["rating_hist_after"] == {}
    json.dumps(summary)  # must not raise / emit invalid tokens
