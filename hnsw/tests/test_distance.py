import numpy as np

from hnsw.utils import l2_dist

def test_simple_2D():
    points = np.array([[1, 0], [1, 1], [4, 0]], dtype=np.float32)
    query = np.array([[1, 0], [1, 0]], dtype=np.float32)

    result = l2_dist(points=points, query=query)

    expected = np.array([[0, 1, 3], [0, 1, 3]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hnsw.utils import l2_dist

# tolarance values are derived from float32 decimal points (roughly)
RTOL = 1e-4
ATOL = 1e-5

finite_floats = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32
)

@st.composite
def points_and_query(draw, min_n=10, max_n=50, min_m=5, max_m=10, min_d=1, max_d=200):
    d = draw(st.integers(min_value=min_d, max_value=max_d))
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    m = draw(st.integers(min_value=min_m, max_value=max_m))
    points = draw(
        arrays(dtype=np.float32, shape=(n, d), elements=finite_floats)
    )
    query = draw(arrays(dtype=np.float32, shape=(m ,d), elements=finite_floats))
    return points, query


class TestL2DistProperties:
    @given(data=points_and_query())
    @settings(max_examples=100)
    def test_matches_oracle(self, data):
        points, query = data

        result = l2_dist(points=points, query=query)
        expected = np.linalg.norm(points - query[:, np.newaxis], axis=2)

        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    @given(data=points_and_query(min_n=1, max_n=20))
    @settings(max_examples=50)
    def test_non_negative(self, data):
        points, query = data

        result = l2_dist(points=points, query=query)

        assert np.all(result >= 0)

    @given(data=points_and_query(min_n=1, max_n=20))
    @settings(max_examples=50)
    def test_distance_to_self_is_zero(self, data):
        points, query = data
        # overwrite one row with the query itself
        points = points.copy()
        points[0] = query[0]

        result = l2_dist(points=points, query=query)

        assert result[0][0] == 0.0

    @given(data=points_and_query(min_n=1, max_n=20))
    @settings(max_examples=50)
    def test_output_shape_and_dtype(self, data):
        points, query = data

        result = l2_dist(points=points, query=query)

        assert result.shape == (query.shape[0], points.shape[0])
