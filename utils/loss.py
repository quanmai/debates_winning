import torch
from torch.nn.modules.loss import _Loss

class PairBCELoss(_Loss):
    """
    Pairwise Logistic function
    """
    def __init__(self):
        super(PairBCELoss, self).__init__()

    def forward(self, pos_logit, neg_logit):
        loss = torch.log(1 + torch.exp(- pos_logit + neg_logit))
        return torch.mean(loss)

class PairHingeLoss(_Loss):
    """ Pairwise Hinge Loss """
    def __init__(self):
        super(PairHingeLoss, self).__init__()

    def forward(self, pos_logit, neg_logit):
        loss = torch.maximum(torch.tensor(0), 1 - pos_logit + neg_logit)
        return torch.mean(loss)