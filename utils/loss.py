import torch
from torch.nn.modules.loss import _Loss

class PairBCELoss_bak(_Loss):
    """ Pairwise Logistic function """
    def __init__(self):
        super(PairBCELoss_bak, self).__init__()

    def forward(self, pos_logit, neg_logit):
        z = pos_logit - neg_logit
        loss = torch.log(1 + torch.exp(-z))
        return torch.mean(loss)
    
class PairBCELoss(_Loss):
    """ Pairwise Logistic function """
    def __init__(self):
        super(PairBCELoss, self).__init__()

    def forward(self, score1, score2, y):
        pos_logit = score1 * y
        neg_logit = score2 * y
        z = pos_logit - neg_logit
        loss = torch.log(1 + torch.exp(-z))
        return torch.mean(loss)

class PairHingeLoss(_Loss):
    """ Pairwise Hinge Loss """
    def __init__(self):
        super(PairHingeLoss, self).__init__()

    def forward(self, pos_logit, neg_logit):
        z = pos_logit - neg_logit
        loss = torch.maximum(torch.tensor(0), 1 - z)
        return torch.mean(loss)