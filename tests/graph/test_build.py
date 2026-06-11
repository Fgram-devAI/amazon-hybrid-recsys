"""Train-only bipartite graph: index maps, propagation/label split, two views."""

import networkx as nx
import pandas as pd

from src.graph.build import BipartiteTrainGraph, build_graph, build_train_bipartite_graph


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


# NOTE: the test cases below are appended in Task 2 of feat/graph-eda-community.

def _toy_eda_train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i2", "rating": 4.0, "timestamp": 2},
            {"user_id": "u2", "parent_asin": "i1", "rating": 3.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "i3", "rating": 5.0, "timestamp": 4},
            {"user_id": "u3", "parent_asin": "i2", "rating": 2.0, "timestamp": 5},
        ]
    )


def test_build_train_graph_returns_bipartite_graph_with_index_maps() -> None:
    result = build_train_bipartite_graph(_toy_eda_train())

    assert isinstance(result, BipartiteTrainGraph)
    assert isinstance(result.graph, nx.Graph)
    assert set(result.user_to_idx) == {"u1", "u2", "u3"}
    assert set(result.item_to_idx) == {"i1", "i2", "i3"}
    assert sorted(result.user_to_idx.values()) == [0, 1, 2]
    assert sorted(result.item_to_idx.values()) == [0, 1, 2]
    for user, idx in result.user_to_idx.items():
        assert result.idx_to_user[idx] == user
    for item, idx in result.item_to_idx.items():
        assert result.idx_to_item[idx] == item


def test_build_train_graph_edges_match_train_rows_with_rating_attribute() -> None:
    result = build_train_bipartite_graph(_toy_eda_train())
    g = result.graph
    assert g.number_of_edges() == 5
    assert g["u1"]["i1"]["rating"] == 5.0
    assert g["u3"]["i2"]["rating"] == 2.0
    assert g.nodes["u1"]["bipartite"] == 0
    assert g.nodes["i1"]["bipartite"] == 1


def test_build_train_graph_ignores_any_test_dataframe_completely() -> None:
    """Sanity: passing only the train frame must not pull in test edges."""
    train = _toy_eda_train()
    test_only = pd.DataFrame(
        [{"user_id": "u_test", "parent_asin": "i_test", "rating": 5.0, "timestamp": 99}]
    )
    result = build_train_bipartite_graph(train)
    assert "u_test" not in result.user_to_idx
    assert "i_test" not in result.item_to_idx
    assert len(test_only) == 1


def test_build_train_graph_deduplicates_repeated_user_item_pairs() -> None:
    repeated = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 4.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 2},
        ]
    )
    result = build_train_bipartite_graph(repeated)
    assert result.graph.number_of_edges() == 1
    assert result.graph["u1"]["i1"]["rating"] == 5.0
