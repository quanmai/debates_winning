import numpy as np
import torch


def top_k_sparsify(A:np.ndarray, k:int) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    adj = np.ones_like(A)
    num_nodes = A.shape[1]
    row_index = np.arange(num_nodes)
    adj[A.argsort(axis=0)[:num_nodes-k], row_index] = 0
    return adj

def acc_score(y_true, y_pred):
    total = 0
    for pred, label in zip(y_pred, y_true):
       total += np.sum(  (pred > 0.5) == (label > 0.5)  ) / len(pred)
    return total / len(y_pred)