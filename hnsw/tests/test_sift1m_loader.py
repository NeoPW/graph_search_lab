import struct
from pathlib import Path

import numpy as np
import pytest

from hnsw.utils import SIFT1MLoader


def write_fvecs(path: Path, vectors: np.typing.NDArray[np.float32]) -> None:
    """Write vectors in the .fvecs format: per-row [int32 dim][float32 * dim]."""
    n, d = vectors.shape
    with path.open("wb") as f:
        for row in vectors:
            f.write(struct.pack("<i", d))
            f.write(row.astype(np.float32).tobytes())


def write_ivecs(path: Path, vectors: np.typing.NDArray[np.int32]) -> None:
    """Write vectors in the .ivecs format: per-row [int32 dim][int32 * dim]."""
    n, d = vectors.shape
    with path.open("wb") as f:
        for row in vectors:
            f.write(struct.pack("<i", d))
            f.write(row.astype(np.int32).tobytes())


@pytest.fixture
def sift_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestSIFT1MLoaderBase:
    def test_load_sift1m_base_roundtrip_values(self, sift_dir: Path) -> None:
        expected = np.array(
            [[1.5, -2.25, 3.0], [0.0, 100.125, -7.5]], dtype=np.float32
        )
        write_fvecs(sift_dir / "sift_base.fvecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_base()

        np.testing.assert_array_equal(result, expected)

    def test_load_sift1m_base_shape(self, sift_dir: Path) -> None:
        expected = np.random.rand(10, 128).astype(np.float32)
        write_fvecs(sift_dir / "sift_base.fvecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_base()

        assert result.shape == (10, 128)

    def test_load_sift1m_base_dtype_is_float32(self, sift_dir: Path) -> None:
        expected = np.random.rand(5, 8).astype(np.float32)
        write_fvecs(sift_dir / "sift_base.fvecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_base()

        assert result.dtype == np.float32

    def test_load_sift1m_base_uses_view_not_cast(self, sift_dir: Path) -> None:
        """
        Regression guard: if someone swaps .view(np.float32) for
        .astype(np.float32), values will be silently wrong because the
        underlying bytes get numerically converted instead of reinterpreted.

        Chosen values include fractions and negatives with bit patterns
        that do NOT round-trip cleanly through int32<->float32 numeric
        conversion, so a wrong implementation reliably fails this test.
        """
        expected = np.array(
            [[3.14159, -0.001, 42.42], [1e10, -1e-10, 123456.789]],
            dtype=np.float32,
        )
        write_fvecs(sift_dir / "sift_base.fvecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_base()

        np.testing.assert_array_equal(result, expected)

    def test_load_sift1m_base_strips_header_column(self, sift_dir: Path) -> None:
        """
        Regression guard for the off-by-one header bug: if the dimension
        header isn't stripped, the array would have d+1 columns, or the
        values would be shifted by one column across every row.
        """
        d = 4
        expected = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        write_fvecs(sift_dir / "sift_base.fvecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_base()

        assert result.shape == (2, d)
        # if header leaked into column 0, this row would start with 4.0 (the
        # dim), not 1.0
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 4.0])


class TestSIFT1MLoaderGroundtruth:
    def test_load_sift1m_gt_roundtrip_values(self, sift_dir: Path) -> None:
        expected = np.array([[3, 7, 1, 9], [0, 2, 5, 8]], dtype=np.int32)
        write_ivecs(sift_dir / "sift_groundtruth.ivecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_gt()

        np.testing.assert_array_equal(result, expected)

    def test_load_sift1m_gt_dtype_is_int32(self, sift_dir: Path) -> None:
        expected = np.array([[1, 2, 3]], dtype=np.int32)
        write_ivecs(sift_dir / "sift_groundtruth.ivecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_gt()

        assert result.dtype == np.int32

    def test_load_sift1m_gt_strips_header_column(self, sift_dir: Path) -> None:
        d = 5
        expected = np.array([[10, 20, 30, 40, 50]], dtype=np.int32)
        write_ivecs(sift_dir / "sift_groundtruth.ivecs", expected)

        loader = SIFT1MLoader(sift_dir)
        result = loader.load_sift1m_gt()

        assert result.shape == (1, d)
        np.testing.assert_array_equal(result[0], [10, 20, 30, 40, 50])