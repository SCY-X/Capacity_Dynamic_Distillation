import torch.nn.functional as F
import torch
import torch.nn as nn
import copy

class KL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, outputs, targets):
        p_s = F.log_softmax(outputs/self.T, dim=1)
        p_t = F.softmax(targets/self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / outputs.shape[0]
        return loss


def Fitnet_loss(feat, margin=None):
    student_feat = feat[0]
    teacher_feat = feat[1]
    length = len(student_feat)
    loss = 0.0
    for i in range(length):
        diff_mat = teacher_feat[i] - student_feat[i]
        l2_loss = torch.norm(diff_mat, p=2, dim=1)
        loss = loss + l2_loss.mean()
    return loss