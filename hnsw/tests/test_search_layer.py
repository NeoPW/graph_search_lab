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

QUERY_NEAR_A = np.array([0.2, 0.2], dtype=np.float32)   # NN = 0
QUERY_NEAR_B = np.array([10.2, 10.2], dtype=np.float32)  # NN = 4


@pytest.fixture
def hnsw():
    return HNSW(layer_graphs=[LAYER_0], top_level=0, entry_points=[0], POINTS=POINTS)


class TestBasicNearestNeighbor:
    """k=1 from various entry points; the core correctness contract."""

    def test_from_far_entry_point_must_cross_bridge(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=1, level=0)
        assert found == [0]

    def test_from_optimal_entry_point_terminates_immediately(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[0], k=1, level=0)
        assert found == [0]

    def test_symmetric_case_query_near_cluster_b(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_B, entry_points=[0], k=1, level=0)
        assert found == [4]

    def test_entry_point_on_the_bridge_itself(self, hnsw):
        # starting exactly at the bridge node should still resolve correctly
        found_a = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[3], k=1, level=0)
        found_b = hnsw.search_layer(query=QUERY_NEAR_B, entry_points=[3], k=1, level=0)
        assert found_a == [0]
        assert found_b == [4]


class TestKParameter:
    """Behavior as k varies, including boundary values."""

    def test_k3_returns_three_closest(self, hnsw):
        # true 3 closest to (0.2, 0.2): 0, then 1 and 2 (tied at dist 0.68)
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=3, level=0)
        assert set(found) == {0, 1, 2}
        assert len(found) == 3

    def test_k_larger_than_total_graph_size(self, hnsw):
        # k=10 but only 8 nodes exist total -> should just return everything
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=10, level=0)
        assert set(found) == set(range(8))
        assert len(found) == 8

    def test_k_equal_to_graph_size(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=8, level=0)
        assert set(found) == set(range(8))

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_found_size_matches_k_when_enough_nodes_reachable(self, hnsw, k):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=k, level=0)
        assert len(found) == k


class TestEntryPoints:
    """Multiple / redundant entry points, and the ef-trimming invariant."""

    def test_multiple_entry_points_trimmed_to_k(self, hnsw):
        # entry_points has 2 elements but k=1: `found` must be trimmed to
        # size 1, not left at size len(entry_points)
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7, 4], k=1, level=0)
        assert found == [0]
        assert len(found) == 1

    def test_entry_points_exceeding_k_are_trimmed_to_closest(self, hnsw):
        # 3 entry points, k=1: only the closest of the 3 may seed `found`,
        # but all 3 should still be explorable via `candidates`
        found = hnsw.search_layer(
            query=QUERY_NEAR_A, entry_points=[5, 6, 7], k=1, level=0
        )
        assert found == [0]

    def test_duplicate_entry_points(self, hnsw):
        # same node passed twice shouldn't break visited-tracking or
        # produce duplicate entries in found
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7, 7], k=1, level=0)
        assert found == [0]

    def test_single_node_graph_as_entry_and_only_reachable_node(self, hnsw):
        # degenerate case: entry point has no way to reach anything better,
        # already-optimal within its own component
        found = hnsw.search_layer(query=QUERY_NEAR_B, entry_points=[7], k=1, level=0)
        assert found == [4]


class TestTiesAndDeterminism:
    """Equidistant candidates shouldn't cause flaky or ill-defined results."""

    def test_tied_distances_both_included_at_k2(self, hnsw):
        # nodes 1 and 2 are exactly tied in distance to query_A
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=2, level=0)
        assert len(found) == 2
        assert 0 in found
        assert len(set(found) & {1, 2}) == 1

    def test_query_equidistant_from_entry_points(self, hnsw):
        # a query exactly between the two bridge nodes should not crash or
        # behave inconsistently regardless of which entry point "wins" first
        midpoint_query = np.array([5.5, 5.5], dtype=np.float32)
        found = hnsw.search_layer(
            query=midpoint_query, entry_points=[3, 4], k=1, level=0
        )
        assert found in ([3], [4])  # either is a valid nearest here


class TestReturnType:
    """Sanity on the shape/type of what search_layer hands back."""

    def test_returns_a_list_of_ints(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=1, level=0)
        assert isinstance(found, list)
        assert all(isinstance(n, (int, np.integer)) for n in found)

    def test_no_duplicate_nodes_in_result(self, hnsw):
        found = hnsw.search_layer(query=QUERY_NEAR_A, entry_points=[7], k=5, level=0)
        assert len(found) == len(set(found))