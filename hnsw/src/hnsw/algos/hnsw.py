# main code for hnsw algo, after paper: https://arxiv.org/abs/1603.09320
# Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824-836, 1 April 2020, doi: 10.1109/TPAMI.2018.2889473.

from hnsw.utils import l2_dist_rank
import numpy as np
import math
import random

class HNSW():
    def __init__(self, layer_graphs: list[dict[int, list[int]]], entry_points: list[int], top_level: int, POINTS: np.typing.NDArray[np.float32]) -> None:
        self.layer_graphs = layer_graphs
        self.entry_points = entry_points #actual entry point getting
        self.top_level = top_level
        self.POINTS = POINTS

    @classmethod
    def empty(cls, POINTS: np.typing.NDArray[np.float32]) -> "HNSW":
        """Construct a fresh, empty HNSW index over the given point set.
        No layers exist yet; the first insert() call will create layer 0
        and establish the inserted element as the initial entry point."""
        return cls(
            layer_graphs=[],
            entry_points=[],
            top_level=-1,
            POINTS=POINTS,
        )

    def k_nn_search(self, query: np.typing.NDArray[np.float32], k: int, cand_list_size: int) -> list[int]:
        closest = []
        entry_points = self.entry_points
        top_level = self.top_level

        for level in range(top_level, 0, -1):
            closest = self.search_layer(
                query=query,
                entry_points=entry_points,
                k=1,
                level=level
            )
            entry_points = [self._get_closest(query=query, points=closest)]

        closest = self.search_layer(
            query=query,
            entry_points=entry_points,
            k=cand_list_size,
            level=0
            )
        return self._top_k_by_distance(query=query, points=closest, k=k)

    # insert new element in layer graphs
    def insert(self, new_element: int, established_connections_num: int, max_conn_per_el_per_layer: list[int], cand_list_size: int, norm_factor: float) -> None:
        closest = []
        entry_points = self.entry_points
        top_level = self.top_level
        new_el_level = math.floor(-math.log(random.uniform(0.0, 1.0)) * norm_factor)

        while len(self.layer_graphs) <= new_el_level:
            self.layer_graphs.append({})
        for level in range(0, new_el_level + 1):
            self.layer_graphs[level].setdefault(new_element, [])

        for level in range(top_level, new_el_level, -1):
            closest = self.search_layer(
                query=self.POINTS[new_element],
                entry_points=entry_points,
                k=cand_list_size,
                level=level
            )
            entry_points = [self._get_closest(query=self.POINTS[new_element], points=closest)]

        for level in range(min(top_level, new_el_level), -1, -1):
            closest = self.search_layer(
                query=self.POINTS[new_element],
                entry_points=entry_points,
                k=cand_list_size,
                level=level
            )
            neighbours = self.select_neigh_simple(query=self.POINTS[new_element], candidates=closest, k=established_connections_num)

            # add conn from new el to neigh
            self.layer_graphs[level][new_element] = neighbours.copy()
            for neigh in neighbours:
                # add reverse edge
                self.layer_graphs[level][neigh].append(new_element)

                # shrink if needed
                if len(self.layer_graphs[level][neigh]) > max_conn_per_el_per_layer[level]:
                    shrunk_connections = self.select_neigh_simple(query=self.POINTS[neigh], candidates=self.layer_graphs[level][neigh], k=max_conn_per_el_per_layer[level])
                    removed = list(set(self.layer_graphs[level][neigh]) - set(shrunk_connections))
                    self.layer_graphs[level][neigh] = shrunk_connections
                    # should only ever remove 1 edge, so 0 as index is fine but needs to be tested (this removes the reverse edge)
                    for r in removed:
                        self.layer_graphs[level][r].remove(neigh)

            entry_points = closest

        if new_el_level > top_level:
            self.top_level = new_el_level
            self.entry_points = [new_element]

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

    def select_neigh_simple(self, query: np.typing.NDArray[np.float32], candidates: list[int], k: int):
        return self._top_k_by_distance(query=query, points=candidates, k=k)

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