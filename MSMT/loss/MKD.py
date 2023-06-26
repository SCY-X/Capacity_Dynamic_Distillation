import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class MKD(nn.Module):
    def __init__(self, p, alpha, beta, mode='L2'):
        super(MKD, self).__init__()
        self.p = p
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

    def forward(self, g_s, g_t):
        return sum([self.mkd_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def mkd_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        batch_size = f_s.size(0)
        f_s = self.at(f_s)
        f_t = self.at(f_t)
        tt_dist_mat = euclidean_dist(f_t.float(), f_t.float())
        st_dist_mat = euclidean_dist(f_s.float(), f_t.float())

        dist = (tt_dist_mat-st_dist_mat).pow(2)
        main_loss = self.alpha * (torch.eye(batch_size).cuda() * dist).sum() / batch_size
        extra_loss = self.beta * ((1.0 - torch.eye(batch_size).cuda()) * dist).sum() / batch_size

        return main_loss + extra_loss

    def at(self, f):
        if self.mode == 'cosine':
            return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
        else:
            return f.pow(self.p).mean(1).view(f.size(0), -1)