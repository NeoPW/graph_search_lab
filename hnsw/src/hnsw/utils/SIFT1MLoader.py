import numpy as np
from pathlib import Path

class SIFT1MLoader():
    def __init__(self, sift1m_dir: Path):
        self.sift1m_dir = sift1m_dir

    def load_sift1m_gt(self) -> np.typing.NDArray[np.int32]:
        path = self.sift1m_dir / "sift_groundtruth.ivecs"
        return self._read_ivecs(path)

    def load_sift1m_base(self) -> np.typing.NDArray[np.float32]:
        path = self.sift1m_dir / "sift_base.fvecs"
        return self._read_fvecs(path)

    def load_sift1m_learn(self) -> np.typing.NDArray[np.float32]:
        path = self.sift1m_dir / "sift_learn.fvecs"
        return self._read_fvecs(path)

    def load_sift1m_query(self) -> np.typing.NDArray[np.float32]:
        path = self.sift1m_dir / "sift_query.fvecs"
        return self._read_fvecs(path)

    # relies on uniform dimension in the data as described here: http://corpus-texmex.irisa.fr/
    def _read_fvecs(self, path: Path) -> np.typing.NDArray[np.float32]:
        a = np.fromfile(path, dtype=np.int32)
        d = a[0]          
        a = a.reshape(-1, d + 1)         
        vectors = a[:, 1:].copy()   
        return vectors.view(np.float32)

    def _read_ivecs(self, path: Path) -> np.typing.NDArray[np.int32]:
        a = np.fromfile(path, dtype=np.int32)
        d = a[0]
        a = a.reshape(-1, d + 1)
        return a[:, 1:].copy()