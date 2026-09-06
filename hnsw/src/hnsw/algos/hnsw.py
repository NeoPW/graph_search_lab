# main code for hnsw algo, after paper: https://arxiv.org/abs/1603.09320
# Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824-836, 1 April 2020, doi: 10.1109/TPAMI.2018.2889473.

from hnsw.utils import l2_dist_rank
import numpy as np

class HNSW():
    def __init__(self, layer_graphs: list[dict[int, list[int]]], POINTS: np.typing.NDArray[np.float32]) -> None:
        self.layer_graphs = layer_graphs
        self.POINTS = POINTS

    # return list of k clostest neighbours in layer_number layer to query, starts greedy search from entry_points
    def search_layer(self, query: np.typing.NDArray[np.float32], entry_points: list[int], k: int, level: int) -> list[int]:
        visited = entry_points.copy()
        candidates = entry_points.copy()
        found = self._top_k_by_distance(query=query, points=entry_points, k=k)

        while len(candidates) > 0:
            candidate = self._get_closest(query=query, points=candidates)
            candidates.remove(candidate)
            furthest = self._get_furthest(query=query, points=found)
            if l2_dist_rank(query=query, points=self.POINTS[candidate]).squeeze() > l2_dist_rank(query=query, points=self.POINTS[furthest]).squeeze():
                break # candidate further away then furthest so we are done

            for neighbour in self.layer_graphs[level][candidate]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    furthest = self._get_furthest(query=query, points=found)

                    if l2_dist_rank(query=query, points=self.POINTS[neighbour]).squeeze() < l2_dist_rank(query=query, points=self.POINTS[furthest]).squeeze() or len(found) < k:
                        candidates.append(neighbour)
                        found.append(neighbour)

                        if len(found) > k:
                            found.remove(furthest)

        return found

    def _get_closest(self, query: np.typing.NDArray[np.float32], points: list[int]) -> int:
        dist = l2_dist_rank(query=query, points=self.POINTS[points])
        return points[int(np.argmin(dist))]

    def _get_furthest(self, query: np.typing.NDArray[np.float32], points: list[int]) -> int:
        dist = l2_dist_rank(query=query, points=self.POINTS[points])
        return points[int(np.argmax(dist))]

    def _top_k_by_distance(self, query: np.typing.NDArray[np.float32], points: list[int], k: int) -> list[int]:
        if len(points) == 0:
            return []
        dist = l2_dist_rank(query=query, points=self.POINTS[points])
        dist = np.atleast_1d(dist.squeeze())
        order = np.argsort(dist)
        k = min(k, len(points))
        return [points[i] for i in order[:k]]