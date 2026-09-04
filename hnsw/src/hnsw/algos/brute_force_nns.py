import numpy as np
from hnsw.utils import l2_dist_rank

# simple brute force approach as easy baseline
# just calc the distance (in this case L2 -> could be made selectable later) between all points & query, then find min by iterating over it
def brute_force_k_nns(k, points, query):
    dist = l2_dist_rank(points=points, query=query)

    kth = min(k, dist.shape[1]) - 1  # -> k > n seems weird
    idx = np.argpartition(dist, kth, axis=1)[:, :k]

    row_idx = np.arange(dist.shape[0])[:, None]
    order = np.argsort(dist[row_idx, idx], axis=1)
    return idx[row_idx, order]