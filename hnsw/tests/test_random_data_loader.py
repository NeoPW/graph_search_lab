import numpy as np
import pytest

from hnsw.data import RandomDataLoader  # adjust to your module / class name


@pytest.fixture
def gen():
    return RandomDataLoader()


def _cluster_ids(zero_std_data):
    """With cluster_std=0 every point sits exactly on its center,
    so identical rows <=> same cluster. Recovers labels without the
    generator having to return them."""
    _, ids, counts = np.unique(zero_std_data, axis=0, return_inverse=True, return_counts=True)
    return ids.ravel(), counts


# ---------------- generate_random_data ----------------

def test_random_shape_and_dtype(gen):
    data = gen.generate_random_data(size=100, dim=8, seed=0)
    assert data.shape == (100, 8)
    assert data.dtype == np.float32


def test_random_values_in_unit_interval(gen):
    data = gen.generate_random_data(size=1000, dim=8, seed=0)
    assert np.all(data >= 0.0) and np.all(data < 1.0)


def test_random_same_seed_reproducible(gen):
    a = gen.generate_random_data(size=100, dim=8, seed=42)
    b = gen.generate_random_data(size=100, dim=8, seed=42)
    np.testing.assert_array_equal(a, b)


def test_random_different_seeds_differ(gen):
    a = gen.generate_random_data(size=100, dim=8, seed=1)
    b = gen.generate_random_data(size=100, dim=8, seed=2)
    assert not np.array_equal(a, b)


# ---------------- generate_clustered_random_data: contract ----------------

def test_clustered_shape_and_dtype(gen):
    data = gen.generate_clustered_random_data(size=100, dim=8, clusters=5, seed=0)
    assert data.shape == (100, 8)
    assert data.dtype == np.float32


def test_clustered_same_seed_reproducible(gen):
    a = gen.generate_clustered_random_data(size=100, dim=8, clusters=5, seed=42)
    b = gen.generate_clustered_random_data(size=100, dim=8, clusters=5, seed=42)
    np.testing.assert_array_equal(a, b)


def test_clustered_different_seeds_differ(gen):
    a = gen.generate_clustered_random_data(size=100, dim=8, clusters=5, seed=1)
    b = gen.generate_clustered_random_data(size=100, dim=8, clusters=5, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("clusters", [0, -1, 101])
def test_clustered_rejects_invalid_cluster_count(gen, clusters):
    with pytest.raises(ValueError):
        gen.generate_clustered_random_data(size=100, dim=8, clusters=clusters, seed=0)


@pytest.mark.parametrize("clusters", [1, 100])
def test_clustered_accepts_boundary_cluster_counts(gen, clusters):
    data = gen.generate_clustered_random_data(size=100, dim=8, clusters=clusters, seed=0)
    assert data.shape == (100, 8)


# ---------------- generate_clustered_random_data: structure (via cluster_std=0) ----------------

@pytest.mark.parametrize("size, clusters", [(100, 5), (103, 5), (50, 50), (50, 1)])
def test_clustered_exact_cluster_count_and_near_equal_sizes(gen, size, clusters):
    data = gen.generate_clustered_random_data(size=size, dim=8, clusters=clusters, seed=0, cluster_std=0.0)
    _, counts = _cluster_ids(data)
    assert len(counts) == clusters
    assert counts.sum() == size
    assert counts.max() - counts.min() <= 1


def test_clustered_output_is_not_grouped_by_cluster(gen):
    size, clusters = 200, 5
    data = gen.generate_clustered_random_data(size=size, dim=8, clusters=clusters, seed=0, cluster_std=0.0)
    ids, _ = _cluster_ids(data)
    # contiguous blocks would change label exactly clusters-1 times
    assert np.count_nonzero(np.diff(ids)) > clusters - 1
    # round-robin assignment would repeat with period `clusters`
    assert not np.array_equal(ids[:-clusters], ids[clusters:])


def test_cluster_std_only_scales_noise(gen):
    # Contract: for a fixed seed, cluster_std changes spread only, never the
    # centers or the assignment. This is what makes a sigma sweep a
    # controlled variable, and what the isolation test below relies on.
    kw = dict(size=200, dim=8, clusters=5, seed=0)
    base = gen.generate_clustered_random_data(**kw, cluster_std=0.0)
    a = gen.generate_clustered_random_data(**kw, cluster_std=0.01)
    b = gen.generate_clustered_random_data(**kw, cluster_std=0.03)
    np.testing.assert_allclose(b - base, 3 * (a - base), atol=1e-6)


# ---------------- fitness for the experiment: isolation ----------------

def test_isolated_regime_is_isolated(gen):
    # Strong geometric isolation for one fixed parameter set:
    # the largest intra-cluster distance is below the smallest inter-cluster distance.
    kw = dict(size=500, dim=10, clusters=10, seed=0)
    ids, _ = _cluster_ids(gen.generate_clustered_random_data(**kw, cluster_std=0.0))
    data = gen.generate_clustered_random_data(**kw, cluster_std=0.01)

    sq = ((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=-1)
    same = ids[:, None] == ids[None, :]
    assert sq[same].max() < sq[~same].min()