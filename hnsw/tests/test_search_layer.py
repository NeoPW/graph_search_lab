import numpy as np
from hnsw.algos import HNSW

points = np.array([
    [0.0, 0.0],   # 0
    [1.0, 0.0],   # 1
    [0.0, 1.0],   # 2
    [1.0, 1.0],   # 3  <- bridge node, cluster A side
    [10.0, 10.0], # 4  <- bridge node, cluster B side
    [11.0, 10.0], # 5
    [10.0, 11.0], # 6
    [11.0, 11.0], # 7
], dtype=np.float32)

layer_0 = {
    0: [1, 2, 3],
    1: [0, 3],
    2: [0, 3],
    3: [0, 1, 2, 4],   # bridge edge 3-4
    4: [3, 5, 6, 7],   # bridge edge on the other side
    5: [4, 7],
    6: [4, 7],
    7: [4, 5, 6],
}

query_A = np.array([0.2, 0.2], dtype=np.float32)  # near cluster A, true NN = 0


def make_hnsw():
    return HNSW(layer_graphs=[layer_0], POINTS=points)


def test_search_layer_k1_from_far_entry_point():
    # entry point is farthest node (cluster B); must cross the bridge
    hnsw = make_hnsw()
    found = hnsw.search_layer(query=query_A, entry_points=[7], k=1, level=0)
    assert found == [0]


def test_search_layer_k1_from_optimal_entry_point():
    # entry point already is the true nearest neighbor
    hnsw = make_hnsw()
    found = hnsw.search_layer(query=query_A, entry_points=[0], k=1, level=0)
    assert found == [0]


def test_search_layer_k3_returns_three_closest():
    # true 3 closest to (0.2, 0.2): 0, then 1 and 2 (tied at dist 0.68)
    hnsw = make_hnsw()
    found = hnsw.search_layer(query=query_A, entry_points=[7], k=3, level=0)
    assert set(found) == {0, 1, 2}
    assert len(found) == 3


def test_search_layer_k_larger_than_reachable_nodes():
    # k=10 but only 8 nodes exist total
    hnsw = make_hnsw()
    found = hnsw.search_layer(query=query_A, entry_points=[7], k=10, level=0)
    assert set(found) == set(range(8))


def test_search_layer_multiple_entry_points():
    hnsw = make_hnsw()
    found = hnsw.search_layer(query=query_A, entry_points=[7, 4], k=1, level=0)
    assert found == [0]