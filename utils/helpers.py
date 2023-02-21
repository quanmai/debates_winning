import numpy as np


def top_k_sparsify(A:np.ndarray, k:int) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    adj = np.ones_like(A)
    num_nodes = A.shape[1]
    row_index = np.arange(num_nodes)
    adj[A.argsort(axis=0)[:num_nodes-k], row_index] = 0
    return adj