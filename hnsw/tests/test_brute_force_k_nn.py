import numpy as np
import pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hnsw.utils import l2_dist, SIFT1MLoader
from hnsw.algos import brute_force_k_nns

RTOL = 1e-4
ATOL = 1e-5

finite_floats = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def points_query_k(draw, min_n=11, max_n=50, min_m=1, max_m=10, min_d=1, max_d=50):
    d = draw(st.integers(min_value=min_d, max_value=max_d))
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    m = draw(st.integers(min_value=min_m, max_value=max_m))
    k = draw(st.integers(min_value=1, max_value=n))
    points = draw(arrays(dtype=np.float32, shape=(n, d), elements=finite_floats))
    query = draw(arrays(dtype=np.float32, shape=(m, d), elements=finite_floats))
    return points, query, k


def test_hand_simple():
    points = np.array([[1, 0], [1, 1], [4, 0], [0, 0]], dtype=np.float32)
    query = np.array([[0, 0], [1, 0]], dtype=np.float32)

    result = brute_force_k_nns(k=2, points=points, query=query)

    assert set(result[0]) == {3, 0}
    assert result[1][0] == 0
    assert set(result[1]) & {1, 3}


class TestBruteForceProperties:
    @given(data=points_query_k())
    @settings(max_examples=100)
    def test_output_shape(self, data):
        points, query, k = data
        result = brute_force_k_nns(k=k, points=points, query=query)
        assert result.shape == (query.shape[0], k)

    @given(data=points_query_k())
    @settings(max_examples=100)
    def test_indices_in_range_and_unique_per_row(self, data):
        points, query, k = data
        result = brute_force_k_nns(k=k, points=points, query=query)
        assert np.all(result >= 0) and np.all(result < points.shape[0])
        for row in result:
            assert len(set(row)) == k  # no duplicate indices within a query's result

    @given(data=points_query_k())
    @settings(max_examples=100)
    def test_matches_full_sort_oracle_by_distance(self, data):
        points, query, k = data
        result = brute_force_k_nns(k=k, points=points, query=query)

        full_dist = l2_dist(points=points, query=query)
        oracle_order = np.argsort(full_dist, axis=1)[:, :k]

        result_dists = np.take_along_axis(full_dist, result, axis=1)
        oracle_dists = np.take_along_axis(full_dist, oracle_order, axis=1)

        # sorting because of ties
        np.testing.assert_allclose(
            np.sort(result_dists, axis=1),
            np.sort(oracle_dists, axis=1),
            rtol=RTOL, atol=ATOL,
        )

    @given(data=points_query_k(min_m=2, max_m=6))
    @settings(max_examples=50)
    def test_batched_matches_looped_single_query_calls(self, data):
        points, query, k = data
        batched = brute_force_k_nns(k=k, points=points, query=query)

        full_dist = l2_dist(points=points, query=query)
        for i in range(query.shape[0]):
            looped = brute_force_k_nns(k=k, points=points, query=query[i:i+1])
            batched_dists = np.sort(full_dist[i][batched[i]])
            looped_dists = np.sort(full_dist[i][looped[0]])
            np.testing.assert_allclose(batched_dists, looped_dists, rtol=RTOL, atol=ATOL)

    def test_k_equals_n_returns_all_points(self):
        points = np.random.default_rng(0).random((7, 3), dtype=np.float64).astype(np.float32)
        query = points[:1].copy()
        result = brute_force_k_nns(k=7, points=points, query=query)
        assert set(result[0]) == set(range(7))


class TestSIFT1MIntegration:
    @pytest.fixture(scope="class")
    def sift_data(self):
        loader = SIFT1MLoader(Path("/home/kldell54304/personal/graph-search-lab/data/sift1m/sift"))
        return loader.load_sift1m_base(), loader.load_sift1m_query(), loader.load_sift1m_gt()

    @pytest.mark.parametrize("n_queries", [1, 2, 50])
    def test_matches_ground_truth_by_distance(self, sift_data, n_queries):
        base, query, gt = sift_data
        q = query[:n_queries]

        calced = brute_force_k_nns(k=100, points=base, query=q)

        dists_expected = l2_dist(query=q, points=base[gt[:n_queries]])
        dists_calced = l2_dist(query=q, points=base[calced])

        np.testing.assert_allclose(
            np.sort(dists_calced, axis=1),
            np.sort(dists_expected, axis=1),
            rtol=RTOL, atol=ATOL,
        )