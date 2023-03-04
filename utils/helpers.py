import numpy as np
import torch


def top_k_sparsify(A:np.ndarray, k: int = 3) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    #TODO: Ugly implemented, have to improve it later!
    # print(f'shape A: {A.shape}')
    adj = np.zeros_like(A)
    prev, nxt = A.shape
    # print(f'prev, nxt: {prev}, {nxt}')
    idx = np.argsort(A, axis=1)
    for i in range(prev):
        for j in range(nxt):
            if idx[i][j] >= nxt - k:
                adj[i][j] = 1
    return adj

def acc_score(y_true, y_pred):
    total = 0
    for pred, label in zip(y_pred, y_true):
       total += np.sum(  (pred > 0.5) == (label > 0.5)  ) / len(pred)
    return total / len(y_pred)