import numpy as np
import torch
import sklearn


# def top_k_sparsify(A:np.ndarray, k: int = 3, is_edge_weight: bool =False) -> np.ndarray:
#     """ Return non-weighted adj matrix given weighted.
#         Edges defined by top-k value.
#     """
#     #TODO: Ugly implemented, have to improve it later!
#     adj = np.zeros_like(A)
#     rows, cols = A.shape
#     idx = np.argsort(A, axis=-1)
#     for r in range(rows):
#         for c in range(cols):
#             if idx[r][c] >= cols - k:
#                 adj[r][c] = 1 #A[r][c] if is_edge_weight else 1
#     return adj

def top_k_sparsify(A:np.ndarray, k: int = 3, is_edge_weight: bool =False) -> np.ndarray:
    """ Return non-weighted adj matrix given weighted.
        Edges defined by top-k value.
    """
    #TODO: Ugly implemented, have to improve it later!
    adj = np.zeros_like(A)
    rows, cols = A.shape
    idx = np.argsort(A, axis=-1)[:,-k:]
    for r in range(rows):
        for i in idx[r]:
            adj[r][i] = A[r][i] if is_edge_weight else 1
    return adj

def threshold_sparsity(A, thres=0.5, is_edge_weight=False, connect_if_isolated=-1):
    """
        Sparsify the adj matrix in thresholding fashion
        i.e., a connection is defined if the edge values >= threshold value
    """
    A = np.where(A>=thres, A, 0) if is_edge_weight else np.where(A>=thres, 1, 0)

    # dealing with isolated node
    # if node connects to no other nodes, connect it with the node that has highest similarity
    rows, cols = A.shape
    for r in range(rows):
        if all(A[r][c]==0 for c in range(cols)):
            A[r][connect_if_isolated] = 1
    return A

def acc_score(y_true, y_pred):
    # print(f'{ y_pred.shape[0]=}')
    return torch.sum(y_true==y_pred) / y_pred.shape[0]

def f1_score(y_true, y_pred):
    y_true = torch.where(y_true <= 0., 0., 1.)
    y_pred = torch.where(y_pred <= 0., 0., 1.)
    # tp = (y_true * y_pred).sum().to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    # fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    # fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    # epsilon = 1e-7
    
    # precision = tp / (tp + fp + epsilon)
    # recall = tp / (tp + fn + epsilon)
    
    # f1 = 2* (precision*recall) / (precision + recall + epsilon)
    # return f1

    from sklearn.metrics import f1_score  
    f1 = f1_score(y_true.cpu().data, y_pred.cpu()) 
    return torch.tensor(f1)