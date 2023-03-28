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

def threshold_sparsity(A, thres=0.5):
    """
        Sparsify the adj matrix in thresholding fashion
        i.e., a connection is defined if the edge values >= threshold value
    """
    A = A.T
    return np.where(A>=thres, 1, 0)

def acc_score(y_true, y_pred):
    return torch.sum(y_true==y_pred) / y_pred.shape[0]