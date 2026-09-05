import numpy as np 

def l2_dist(query: np.typing.NDArray[np.float32], points: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    return np.sqrt(np.sum(np.square(points - query[:, np.newaxis]), 2))

# for ranking purposes the sqrt is not needed
def l2_dist_rank(
    query: np.typing.NDArray[np.float32], points: np.typing.NDArray[np.float32]
) -> np.typing.NDArray[np.float32]:
    """
    returns a 2D array of shape (Q, N)
    """
    q = np.atleast_2d(query)
    p = np.atleast_2d(points)
    return np.sum(np.square(p - q[:, np.newaxis]), axis=2)