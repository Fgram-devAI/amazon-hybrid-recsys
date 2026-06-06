"""Compact EDA summary for the report and pipeline sanity-checking."""


def _n_users(df):
    return int(df["user_id"].nunique())


def _n_items(df):
    return int(df["parent_asin"].nunique())


def _sparsity(df):
    users, items = _n_users(df), _n_items(df)
    if users == 0 or items == 0:
        return None
    return 1.0 - (len(df) / (users * items))


def _rating_hist(df):
    counts = df["rating"].round(1).value_counts().sort_index()
    return {str(rating): int(n) for rating, n in counts.items()}


def summarize_eda(
    raw_count,
    valid_count,
    deduped_df,
    kcore_df,
    train_df,
    test_df,
    min_rating_relevant,
):
    """Build a JSON-serializable dict of dataset counts and distributions."""
    return {
        "raw_reviews": int(raw_count),
        "after_drop_invalid": int(valid_count),
        "after_dedup": len(deduped_df),
        "after_kcore": len(kcore_df),
        "users_before": _n_users(deduped_df),
        "users_after": _n_users(kcore_df),
        "items_before": _n_items(deduped_df),
        "items_after": _n_items(kcore_df),
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "sparsity_before": _sparsity(deduped_df),
        "sparsity_after": _sparsity(kcore_df),
        "rating_hist_before": _rating_hist(deduped_df),
        "rating_hist_after": _rating_hist(kcore_df),
        "pct_relevant_after": float((kcore_df["rating"] >= min_rating_relevant).mean()),
    }
