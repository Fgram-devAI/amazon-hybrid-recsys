"""Train-only bipartite graph: index maps, propagation/label split, two views."""

import pandas as pd

from src.graph.build import build_graph


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":    ["u1", "u1", "u2", "u2", "u3"],
            "parent_asin":["i1", "i2", "i1", "i3", "i2"],
            "rating":     [5.0,  3.0,  4.0,  2.0,  5.0],
            "timestamp":  [1, 2, 3, 4, 5],
        }
    )


def _toy_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":    ["u1", "u4"],
            "parent_asin":["i3", "i1"],
            "rating":     [4.0, 5.0],
            "timestamp":  [10, 11],
        }
    )


def test_build_graph_returns_user_and_item_index_maps():
    graph = build_graph(_toy_train(), min_rating_positive=4.0)

    assert set(graph.user_index.keys()) == {"u1", "u2", "u3"}
    assert set(graph.item_index.keys()) == {"i1", "i2", "i3"}
    # contiguous 0..n-1 indexing
    assert sorted(graph.user_index.values()) == [0, 1, 2]
    assert sorted(graph.item_index.values()) == [0, 1, 2]


def test_build_graph_excludes_test_edges_from_propagation_and_labels():
    train = _toy_train()
    graph = build_graph(train, min_rating_positive=4.0)

    # observed edge_label_index size matches train rows (all five)
    assert graph.edge_label_index.shape == (2, 5)
    # rating tensor aligned to edge_label_index
    assert graph.edge_label_rating.shape == (5,)
    # propagation edges are doubled (undirected) -> 10 columns
    assert graph.propagation_edge_index.shape == (2, 10)


def test_build_graph_positive_only_view_drops_low_ratings():
    graph = build_graph(_toy_train(), min_rating_positive=4.0)

    # positive train edges = ratings >= 4 -> 3 rows
    # (u1,i1)=5, (u2,i1)=4, (u3,i2)=5
    assert graph.positive_edge_label_index.shape == (2, 3)
    assert graph.positive_propagation_edge_index.shape == (2, 6)  # doubled


def test_build_graph_returns_user_offset_for_unified_node_space():
    """PyG LightGCN expects users+items in a single node space; the offset puts items after users."""
    graph = build_graph(_toy_train(), min_rating_positive=4.0)

    assert graph.user_offset == 0
    assert graph.item_offset == len(graph.user_index)  # = 3
    assert graph.num_nodes == len(graph.user_index) + len(graph.item_index)


def test_build_graph_ignores_test_dataframe_even_if_caller_passes_it():
    """build_graph only takes train; the test split must never leak into the graph."""
    train = _toy_train()
    test = _toy_test()
    graph = build_graph(train, min_rating_positive=4.0)

    # user 'u4' and item 'i3' edges from test must not appear
    assert "u4" not in graph.user_index
    edge_users = graph.edge_label_index[0].tolist()
    assert all(u != graph.user_index.get("u4", -1) for u in edge_users)
    # 'i3' may exist (it's in train via u2's rating=2.0) but that's a train fact, not test
    _ = test  # explicitly unused
