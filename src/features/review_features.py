"""Train-only user/item review aggregates.

The leakage rule (spec §4) is enforced by the train DataFrame argument: this
module only joins sentiment scores onto rows that already appear in ``train``,
so a held-out test review can never reach the aggregates.
"""

from __future__ import annotations

import pandas as pd


def _train_sentiment(train: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Left-join train rows with sentiment cache on (user, item, ts); train rows always survive."""
    return train.merge(
        sentiment,
        on=["user_id", "parent_asin", "timestamp"],
        how="left",
    )


def user_review_aggregates(
    train: pd.DataFrame, sentiment: pd.DataFrame
) -> pd.DataFrame:
    """Per-user features from training ratings + train-aligned sentiment."""
    merged = _train_sentiment(train, sentiment)
    grouped = merged.groupby("user_id")
    out = pd.DataFrame(
        {
            "user_id": list(grouped.groups.keys()),
            "user_mean_rating": grouped["rating"].mean().to_numpy(),
            "user_rating_std": grouped["rating"].std(ddof=0).fillna(0.0).to_numpy(),
            "user_review_sentiment_mean": grouped["sentiment_score"]
            .mean()
            .fillna(0.0)
            .to_numpy(),
            "user_review_sentiment_std": grouped["sentiment_score"]
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy(),
        }
    )
    out["user_rating_minus_sentiment_gap"] = (
        out["user_mean_rating"] - out["user_review_sentiment_mean"]
    )
    out["user_strictness_score"] = -out["user_rating_minus_sentiment_gap"]
    return out


def item_review_aggregates(
    train: pd.DataFrame, sentiment: pd.DataFrame
) -> pd.DataFrame:
    """Per-item features from training reviews only; cold items default to 0.0."""
    merged = _train_sentiment(train, sentiment)
    grouped = merged.groupby("parent_asin")
    sentiment_mean = grouped["sentiment_score"].mean().fillna(0.0)
    sentiment_count = grouped["sentiment_score"].count()
    rating_mean = grouped["rating"].mean()
    out = pd.DataFrame(
        {
            "parent_asin": list(grouped.groups.keys()),
            "item_mean_rating": rating_mean.to_numpy(),
            "item_train_sentiment_mean": sentiment_mean.to_numpy(),
            "item_train_sentiment_count": sentiment_count.to_numpy().astype("int64"),
        }
    )
    out["item_rating_minus_sentiment_gap"] = (
        out["item_mean_rating"] - out["item_train_sentiment_mean"]
    )
    return out
