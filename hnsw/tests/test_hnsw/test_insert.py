import math
import random

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


def force_level(monkeypatch, level: int, norm_factor: float = 1.0) -> None:
    """Force new_el_level to come out to exactly `level`, given norm_factor,
    by pinning random.uniform's return value so
    floor(-log(u) * norm_factor) == level exactly."""
    u = math.exp(-level / norm_factor)
    monkeypatch.setattr(random, "uniform", lambda a, b: u)


class TestEmptyGraphFirstInsertion:

    def test_first_insertion_creates_layer_zero_and_entry_point(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        force_level(monkeypatch, level=0)

        hnsw.insert(
            new_element=0, established_connections_num=2,
            max_conn_per_el_per_layer=[4], cand_list_size=4, norm_factor=1.0,
        )

        assert hnsw.top_level == 0
        assert hnsw.entry_points == [0]
        assert 0 in hnsw.layer_graphs[0]
        assert hnsw.layer_graphs[0][0] == []

    def test_first_insertion_at_higher_level_grows_layer_graphs(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        force_level(monkeypatch, level=2)

        hnsw.insert(
            new_element=0, established_connections_num=2,
            max_conn_per_el_per_layer=[4, 4, 4], cand_list_size=4, norm_factor=1.0,
        )

        assert hnsw.top_level == 2
        assert hnsw.entry_points == [0]
        assert len(hnsw.layer_graphs) == 3
        # node must exist (with no neighbors) at every layer 0..2
        for level in range(3):
            assert 0 in hnsw.layer_graphs[level]
            assert hnsw.layer_graphs[level][0] == []


class TestSequentialInsertion:

    def test_second_node_same_level_connects_bidirectionally(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        force_level(monkeypatch, level=0)
        hnsw.insert(0, 2, [4], 4, 1.0)

        force_level(monkeypatch, level=0)
        hnsw.insert(1, 2, [4], 4, 1.0)

        assert 1 in hnsw.layer_graphs[0][0]
        assert 0 in hnsw.layer_graphs[0][1]
        # top_level/entry_points unchanged: new_el_level (0) did not exceed top_level (0)
        assert hnsw.top_level == 0
        assert hnsw.entry_points == [0]

    def test_new_node_at_higher_level_replaces_entry_point(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        force_level(monkeypatch, level=0)
        hnsw.insert(0, 2, [4], 4, 1.0)

        force_level(monkeypatch, level=2)
        hnsw.insert(1, 2, [4, 4, 4], 4, 1.0)

        assert hnsw.top_level == 2
        assert hnsw.entry_points == [1]
        # node 1 must be connected to node 0 at layer 0 (only existing peer)
        assert 0 in hnsw.layer_graphs[0][1]
        assert 1 in hnsw.layer_graphs[0][0]
        # but isolated (no peers) at layers 1 and 2
        assert hnsw.layer_graphs[1][1] == []
        assert hnsw.layer_graphs[2][1] == []

    def test_search_at_sparsely_populated_upper_layer_does_not_raise(self, monkeypatch):
        # regression test: node present at a layer with zero neighbors
        # must not KeyError on lookup
        hnsw = HNSW.empty(POINTS=POINTS)
        force_level(monkeypatch, level=0)
        hnsw.insert(0, 2, [4], 4, 1.0)

        force_level(monkeypatch, level=2)
        hnsw.insert(1, 2, [4, 4, 4], 4, 1.0)

        found = hnsw.search_layer(query=POINTS[0], entry_points=[1], k=1, level=1)
        assert found == [1]

    def test_five_nodes_all_level_zero_all_pairwise_reachable(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        for i in range(5):
            force_level(monkeypatch, level=0)
            hnsw.insert(i, 2, [4], 4, 1.0)

        assert hnsw.top_level == 0
        assert len(hnsw.layer_graphs[0]) == 5
        # every node should be able to find node 0 as its nearest (all clustered together)
        for i in range(1, 5):
            found = hnsw.search_layer(query=POINTS[0], entry_points=[i], k=1, level=0)
            assert 0 in found or i == 0


class TestShrinkAndReverseEdgeRemoval:

    def test_degree_never_exceeds_max_conn_cap(self, monkeypatch):
        hnsw = HNSW.empty(POINTS=POINTS)
        # tight cap of 2 connections per node at layer 0
        for i in range(8):
            force_level(monkeypatch, level=0)
            hnsw.insert(i, 2, [2], 4, 1.0)

        for node, neighbours in hnsw.layer_graphs[0].items():
            assert len(neighbours) <= 2, f"node {node} exceeded cap: {neighbours}"

    def test_shrink_removes_reverse_edge_not_just_forward(self, monkeypatch):
        # construct a case where a node is forced over its cap, and verify
        # the dropped neighbor no longer references the shrunk node back
        hnsw = HNSW.empty(POINTS=POINTS)
        for i in range(8):
            force_level(monkeypatch, level=0)
            hnsw.insert(i, 2, [2], 4, 1.0)

        # for every edge (a, b) in the final graph, b must also list a --
        # i.e. no dangling one-directional edges left over from a shrink
        for node, neighbours in hnsw.layer_graphs[0].items():
            for neighbour in neighbours:
                assert node in hnsw.layer_graphs[0][neighbour], (
                    f"asymmetric edge: {node} -> {neighbour} exists, "
                    f"but {neighbour} -> {node} does not"
                )


class TestDeterminismHelperSanity:

    def test_force_level_produces_expected_level(self, monkeypatch):
        # sanity-check the test helper itself, isolated from insert() entirely
        force_level(monkeypatch, level=3, norm_factor=1.0)
        sampled = math.floor(-math.log(random.uniform(0.0, 1.0)) * 1.0)
        assert sampled == 3

    def test_force_level_respects_norm_factor(self, monkeypatch):
        force_level(monkeypatch, level=2, norm_factor=0.5)
        sampled = math.floor(-math.log(random.uniform(0.0, 1.0)) * 0.5)
        assert sampled == 2