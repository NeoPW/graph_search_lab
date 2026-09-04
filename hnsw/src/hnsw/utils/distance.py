import numpy as np 

def l2_dist(query: np.typing.NDArray[np.float32], points: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    return np.sqrt(np.sum(np.square(points - query[:, np.newaxis]), 2))

# for ranking purposes the sqrt is not needed
def l2_dist_rank(query: np.typing.NDArray[np.float32], points: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    return np.sum(np.square(points - query[:, np.newaxis]), 2)