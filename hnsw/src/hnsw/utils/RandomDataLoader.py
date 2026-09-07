import numpy as np
import random
import math

from hnsw.algos import brute_force_k_nns

class RandomDataLoader:
    def __init__(self, date_set_size: int, query_size: int, dim: int, seed: int):
        self.data_set_size = date_set_size
        self.query_size = query_size
        self.dim = dim
        self.gen = np.random.default_rng(seed=seed)

    def generate_data_set(self) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        data_set = self.gen.random(size=(self.data_set_size, self.dim), dtype=np.float32)
        query_set = self.gen.random(size=(self.query_size, self.dim), dtype=np.float32)

        return data_set, query_set