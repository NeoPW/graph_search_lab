import numpy as np
import random
import math

from hnsw.algos import brute_force_k_nns

class RandomDataLoader:
    def __init__(self):
        pass

    def generate_random_data(self, size: int, dim: int, seed:int) -> np.typing.NDArray[np.float32]:
        gen = np.random.default_rng(seed=seed)
        data_set = gen.random(size=(size, dim), dtype=np.float32)

        return data_set

    def generate_clustered_random_data(self, size: int, dim: int, clusters: int, seed: int,
                                       cluster_std: float = 0.02) -> np.typing.NDArray[np.float32]:
        if not 1 <= clusters <= size:
            raise ValueError(f"need 1 <= clusters <= size, got clusters={clusters}, size={size}")

        gen = np.random.default_rng(seed=seed)

        # cluster centers random uniform points
        centers = gen.random(size=(clusters, dim), dtype=np.float32)

        # near-equal cluster sizes, assigned in shuffled order
        labels = gen.permutation(np.arange(size) % clusters)

        # gaussian noise around base points
        noise = gen.standard_normal(size=(size, dim), dtype=np.float32) * cluster_std
        data_set = centers[labels] + noise

        return data_set.astype(np.float32, copy=False)

    def generate_gt(self, data_set: np.typing.NDArray[np.float32], query_set: np.typing.NDArray[np.float32], k: int):
        return brute_force_k_nns(
            k=k,
            query=query_set,
            points=data_set
        )