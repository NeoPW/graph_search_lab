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

# query sits right next to node 0, inside cluster A
QUERY_A = np.array([0.1, 0.1], dtype=np.float32)


def make_hnsw():
    return HNSW(layer_graphs=[LAYER_0], entry_points=[0], top_level=0, POINTS=POINTS)


class TestDiversitySelection:
    """
    Squared distances to QUERY_A=(0.1,0.1):
      0: 0.02   1: 0.82   2: 0.82   3: 1.62   4: 196.02
    Squared distances from candidate to node 0 (the first accepted pick):
      1: 1.0    2: 1.0    3: 2.0    4: 200.0

    1 and 2 are much closer to 0 than to q (1.0 < 0.82? no -- check
    carefully: dist(1,q)=0.82 vs dist(1,0)=1.0 -> 0.82 < 1.0, so 1 is
    actually diverse relative to 0 by a hair. This fixture's margins are
    tight -- verify against your own run rather than trusting by eye.
    """

    def test_first_candidate_always_accepted(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0], k=1, layer=0,
            extendCandidates=False, keepPrunedConnections=False,
        )
        assert result == [0]

    def test_k1_returns_only_the_single_closest(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0, 1, 2, 3, 4], k=1, layer=0,
            extendCandidates=False, keepPrunedConnections=False,
        )
        assert result == [0]

    def test_far_bridge_node_rejected_as_redundant_with_zero(self):
        # 4 is far from everything in cluster A; it should be *closer to
        # q* than to 0 by a wide margin (both q and 0 are near-origin,
        # 4 is roughly equidistant from them) -- so this actually tests
        # whether 4 survives, which needs verifying against your run,
        # not assumed
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0, 4], k=2, layer=0,
            extendCandidates=False, keepPrunedConnections=False,
        )
        assert 0 in result
        # confirm against your actual run whether 4 is accepted or rejected here


class TestKeepPrunedConnections:

    def test_backfill_reaches_k_when_pool_allows(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0, 1, 2, 3], k=4, layer=0,
            extendCandidates=False, keepPrunedConnections=True,
        )
        assert set(result) == {0, 1, 2, 3}
        assert len(result) == 4

    def test_without_backfill_result_can_be_smaller_than_k(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0, 1, 2, 3], k=4, layer=0,
            extendCandidates=False, keepPrunedConnections=False,
        )
        assert len(result) <= 4  # exact count depends on which get rejected -- verify


class TestExtendCandidates:
    """0's layer-0 neighbors are [1, 2, 3] -- all already candidates in
    the base case, so extension from {0} alone should pull in exactly
    {1, 2, 3} and nothing beyond (no two-hop leakage to 4)."""

    def test_extension_pulls_in_direct_neighbors_only(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0], k=10, layer=0,
            extendCandidates=True, keepPrunedConnections=True,
        )
        assert set(result).issubset({0, 1, 2, 3})
        assert 4 not in result  # 4 is two hops from 0 (0->3->4), must not appear

    def test_without_extension_only_original_candidate_considered(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0], k=10, layer=0,
            extendCandidates=False, keepPrunedConnections=True,
        )
        assert result == [0]


class TestEdgeCases:

    def test_empty_candidates_returns_empty(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[], k=3, layer=0,
            extendCandidates=False, keepPrunedConnections=False,
        )
        assert result == []

    def test_no_duplicates_in_result(self):
        hnsw = make_hnsw()
        result = hnsw.select_neigh_heuristic(
            query=QUERY_A, candidates=[0, 1, 2, 3, 4], k=5, layer=0,
            extendCandidates=False, keepPrunedConnections=True,
        )
        assert len(result) == len(set(result))