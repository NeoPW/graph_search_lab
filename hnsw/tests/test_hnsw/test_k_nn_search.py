import numpy as np
import pytest

from hnsw.algos import HNSW

POINTS = np.array([
    [0.0, 0.0],   # 0
    [1.0, 0.0],   # 1
    [0.0, 1.0],   # 2
    [1.0, 1.0],   # 3  <- bridge node, cluster A side
    [10.0, 10.0], # 4  <- bridge node, cluster B side
    [11.0, 10.0], # 5
    [10.0, 11.0], # 6
    [11.0, 11.0], # 7
], dtype=np.float32)

LAYER_0 = {
    0: [1, 2, 3],
    1: [0, 3],
    2: [0, 3],
    3: [0, 1, 2, 4],   # bridge edge 3-4
    4: [3, 5, 6, 7],   # bridge edge on the other side
    5: [4, 7],
    6: [4, 7],
    7: [4, 5, 6],
}

# sparse upper layer: only the two bridge nodes exist here, giving the
# search a long-range hop before descending into the dense layer-0 graph
LAYER_1 = {
    3: [4],
    4: [3],
}

QUERY_NEAR_A = np.array([0.2, 0.2], dtype=np.float32)     # true NN = 0
QUERY_NEAR_B = np.array([10.2, 10.2], dtype=np.float32)   # true NN = 4


def make_single_layer_hnsw(entry_points):
    return HNSW(
        layer_graphs=[LAYER_0],
        entry_points=entry_points,
        top_level=0,
        POINTS=POINTS,
    )


def make_two_layer_hnsw(entry_points):
    return HNSW(
        layer_graphs=[LAYER_0, LAYER_1],
        entry_points=entry_points,
        top_level=1,
        POINTS=POINTS,
    )


class TestSingleLayerBasics:

    def test_finds_nearest_across_bridge_from_far_entry_point(self):
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=1, cand_list_size=4)
        assert result == [0]

    def test_entry_point_already_optimal(self):
        hnsw = make_single_layer_hnsw(entry_points=[0])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=1, cand_list_size=4)
        assert result == [0]

    def test_k3_returns_three_closest_including_tie(self):
        # true 3 closest to (0.2, 0.2): 0, then 1 and 2 (tied)
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=3, cand_list_size=6)
        assert set(result) == {0, 1, 2}
        assert len(result) == 3

    def test_symmetric_query_near_cluster_b(self):
        hnsw = make_single_layer_hnsw(entry_points=[0])
        result = hnsw.k_nn_search(query=QUERY_NEAR_B, k=1, cand_list_size=4)
        assert result == [4]


class TestMultiLayerDescent:
    """Exercises the one part of k_nn_search that isn't just reused
    search_layer/_top_k_by_distance machinery: the top-down ef=1 descent."""

    def test_descends_through_sparse_top_layer_to_correct_answer(self):
        hnsw = make_two_layer_hnsw(entry_points=[3])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=1, cand_list_size=4)
        assert result == [0]

    def test_descends_and_crosses_to_other_cluster(self):
        hnsw = make_two_layer_hnsw(entry_points=[3])
        result = hnsw.k_nn_search(query=QUERY_NEAR_B, k=1, cand_list_size=4)
        assert result == [4]

    def test_entry_point_at_top_layer_is_the_far_bridge_node(self):
        # entry point 4 is far from query_A in raw distance, but layer 1
        # only contains {3, 4} -- must correctly narrow to 3 before
        # descending, not get stuck reasoning about layer-0 distances early
        hnsw = make_two_layer_hnsw(entry_points=[4])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=1, cand_list_size=4)
        assert result == [0]


class TestCandListSizeBehavior:

    def test_cand_list_size_smaller_than_k_caps_result_length(self):
        # cand_list_size limits the candidate pool search_layer returns at
        # layer 0; if it's smaller than k, there simply aren't k candidates
        # to choose from -- result length should match cand_list_size, not k
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=5, cand_list_size=2)
        assert len(result) == 2

    def test_cand_list_size_larger_than_graph_returns_all_reachable(self):
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=10, cand_list_size=20)
        assert set(result) == set(range(8))
        assert len(result) == 8


class TestReturnType:

    def test_returns_list_of_ints(self):
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=3, cand_list_size=6)
        assert isinstance(result, list)
        assert all(isinstance(n, (int, np.integer)) for n in result)

    def test_no_duplicate_nodes_in_result(self):
        hnsw = make_single_layer_hnsw(entry_points=[7])
        result = hnsw.k_nn_search(query=QUERY_NEAR_A, k=5, cand_list_size=8)
        assert len(result) == len(set(result))