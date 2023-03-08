import numpy as np
import torch


def top_k_sparsify(A:np.ndarray, k: int = 3) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    #TODO: Ugly implemented, have to improve it later!
    A = A.T
    adj = np.zeros_like(A)
    cur, prev = A.shape
    idx = np.argsort(A, axis=-1)
    for i in range(cur):
        for j in range(prev):
            if idx[i][j] >= prev - k:
                adj[i][j] = 1
    return adj

def acc_score(y_true, y_pred):
    total = 0
    # print(f'y_true {y_true}')
    # print(f'y_pred {y_pred}')
    for pred, label in zip(y_pred, y_true):
       total += torch.sum((pred > 0.5) == (label > 0.5)) / len(pred)
    return total / y_pred.shape[0]