import torch
from torch import nn
import torch.nn.functional as F


class BasicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_loss(self, y_pred, b_p, p_l, p_s, t_l, t_s):
        y_pred = F.softmax(y_pred, dim=-1)
        y_true = torch.where(p_l > t_l, 1, 0) + torch.where(p_s > t_s, 2, 0)
        y_true = torch.where(y_true == 3, 0, y_true)

        loss = F.cross_entropy(y_pred, y_true, reduction="none").sum()
        return loss
