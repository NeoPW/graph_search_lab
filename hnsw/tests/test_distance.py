# tests/test_l2_dist_rank.py

import numpy as np
import pytest

from hnsw.utils import l2_dist_rank


def naive_sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Reference implementation: plain squared L2 between two 1D points."""
    return float(np.sum((a - b) ** 2))


class TestShapes:
    def test_single_query_single_point_1d_inputs(self):
        q = np.array([0.0, 0.0], dtype=np.float32)
        p = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_dist_rank(q, p)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(25.0)

    def test_single_query_single_point_already_2d(self):
        q = np.array([[0.0, 0.0]], dtype=np.float32)
        p = np.array([[3.0, 4.0]], dtype=np.float32)
        result = l2_dist_rank(q, p)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(25.0)

    def test_single_query_multiple_points(self):
        q = np.array([0.0, 0.0], dtype=np.float32)
        points = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = l2_dist_rank(q, points)
        assert result.shape == (1, 3)
        expected = np.array([1.0, 4.0, 25.0])
        np.testing.assert_allclose(result[0], expected)

    def test_batch_queries_multiple_points(self):
        # oracle-style: Q queries against N points -> (Q, N)
        queries = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        points = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = l2_dist_rank(queries, points)
        assert result.shape == (2, 3)
        # row 0: distances from (0,0)
        np.testing.assert_allclose(result[0], [1.0, 4.0, 25.0])
        # row 1: distances from (10,10)
        expected_row1 = [
            naive_sq_dist(np.array([10.0, 10.0]), np.array([1.0, 0.0])),
            naive_sq_dist(np.array([10.0, 10.0]), np.array([0.0, 2.0])),
            naive_sq_dist(np.array([10.0, 10.0]), np.array([3.0, 4.0])),
        ]
        np.testing.assert_allclose(result[1], expected_row1)


class TestCorrectness:
    def test_zero_distance_when_equal(self):
        q = np.array([5.0, -3.0], dtype=np.float32)
        p = np.array([5.0, -3.0], dtype=np.float32)
        result = l2_dist_rank(q, p)
        assert result[0, 0] == pytest.approx(0.0)

    def test_matches_naive_loop_implementation(self):
        rng = np.random.default_rng(42)
        queries = rng.random((4, 5)).astype(np.float32)
        points = rng.random((7, 5)).astype(np.float32)

        result = l2_dist_rank(queries, points)

        expected = np.array(
            [[naive_sq_dist(q, p) for p in points] for q in queries]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEdgeCases:
    def test_single_dimension_points(self):
        # d=1: exercises the newaxis broadcasting with the smallest possible d
        q = np.array([2.0], dtype=np.float32)
        points = np.array([[5.0], [1.0]], dtype=np.float32)
        result = l2_dist_rank(q, points)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [9.0, 1.0])

    def test_single_point_array_shape_d_vs_shape_1_d(self):
        # (d,) and (1, d) must be treated identically for a single point
        q = np.array([1.0, 1.0], dtype=np.float32)
        p_flat = np.array([4.0, 5.0], dtype=np.float32)
        p_2d = np.array([[4.0, 5.0]], dtype=np.float32)

        result_flat = l2_dist_rank(q, p_flat)
        result_2d = l2_dist_rank(q, p_2d)
        np.testing.assert_allclose(result_flat, result_2d)

    def test_output_is_always_2d_regardless_of_input_rank(self):
        q = np.array([0.0, 0.0], dtype=np.float32)
        p = np.array([1.0, 1.0], dtype=np.float32)
        result = l2_dist_rank(q, p)
        assert result.ndim == 2