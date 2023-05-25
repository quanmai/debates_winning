import numpy as np
import torch


def top_k_sparsify(A:np.ndarray, k: int = 3) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    #TODO: Ugly implemented, have to improve it later!
    adj = np.zeros_like(A)
    rows, cols = A.shape
    idx = np.argsort(A, axis=-1)
    for r in range(rows):
        for c in range(cols):
            if idx[r][c] >= cols - k:
                adj[r][c] = 1
    return adj

def threshold_sparsity(A, thres=0.5, connect_if_isolated=-1):
    """
        Sparsify the adj matrix in thresholding fashion
        i.e., a connection is defined if the edge values >= threshold value
    """
    A = np.where(A>=thres, 1, 0)

    # dealing with isolated node
    # if node connects to no other nodes, connect it with the node that has highest similarity
    rows, cols = A.shape
    for r in range(rows):
        if all(A[r][c]==0 for c in range(cols)):
            A[r][connect_if_isolated] = 1
    return A

def acc_score(y_true, y_pred):
    return torch.sum(y_true==y_pred) / y_pred.shape[0]

def f1_score(y_true, y_pred):
    # y_true = torch.where(y_true<0, 0, 1)
    # y_pred = torch.where(y_pred<0, 0, 1)
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1