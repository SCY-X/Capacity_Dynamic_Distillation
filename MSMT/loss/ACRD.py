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
#
# def hard_example_mining(dist_mat, labels):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#       dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#       labels: pytorch LongTensor, with shape [N]
#       return_inds: whether to return the indices. Save time if `False`(?)
#     Returns:
#       dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#       dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#       p_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#       n_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples,
#       thus we can cope with all anchors in parallel.
#     """
#
#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)
#
#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
#
#
#
#     dist_ap, relative_p_inds = torch.max(
#             dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#         # `dist_an` means distance(anchor, negative)
#         # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(
#             dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#         # shape [N]
#
#
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)
#     return dist_ap, dist_an


def smooth_rank(distmat, batch_size, mode='l2'):

    # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors

    sim_all_repeat = distmat.unsqueeze(dim=1).repeat(1, batch_size, 1)

    # compute the difference matrix
    sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)

    # pass through the relu
    if mode == 'l2':
        sim_sg = F.relu(-sim_diff, inplace=True)
    else:
        sim_sg = F.relu(sim_diff, inplace=True)

    # compute the rankings

    sim_all_rk = torch.sum(sim_sg, dim=-1)
    return sim_all_rk


class ACRD(nn.Module):
    def __init__(self, p, k, alpha, beta, mode='l2', pool='all'):
        super(ACRD, self).__init__()
        self.p = p
        self.mode = mode
        self.pool = pool
        self.alpha = alpha
        self.beta = beta
        self.teacher_features = []
        self.teacher_features1 = []
        self.k = k


    def forward(self, g_s, g_t, statistics):
        return sum([self.acrd_loss(f_s, f_t, layer, statistics) for layer, (f_s, f_t) in enumerate(zip(g_s, g_t))])

    def acrd_loss(self, f_s, f_t, layer, statistics):
        batch_size = f_s.size(0)
        if self.pool == 'gap':
            f_s = F.adaptive_avg_pool2d(f_s.pow(self.p), 1).view(batch_size, -1)
            f_t = F.adaptive_avg_pool2d(f_t.pow(self.p), 1).view(batch_size, -1)

        elif self.pool == 'cap':
            f_s = f_s.pow(self.p).mean(1).view(f_s.size(0), -1)
            f_t = f_t.pow(self.p).mean(1).view(f_t.size(0), -1)

        else:
            f_s = f_s.pow(self.p).view(f_s.size(0), -1)
            f_t = f_t.pow(self.p).view(f_t.size(0), -1)

        if statistics:
            if layer == 0:
                if len(self.teacher_features) == 0:
                    self.teacher_features = f_t
                else:
                    self.teacher_features = torch.cat((self.teacher_features, f_t), dim=0)

            elif layer == 1:
                if len(self.teacher_features1) == 0:
                    self.teacher_features1 = f_t
                else:
                    self.teacher_features1 = torch.cat((self.teacher_features1, f_t), dim=0)


        if self.mode == 'cosine':
            f_s = F.normalize(f_s, p=2, dim=1)
            f_t = F.normalize(f_t, p=2, dim=1)
            if layer == 0:
                gallery_f = F.normalize(self.teacher_features, p=2, dim=1)
            elif layer == 1:
                gallery_f = F.normalize(self.teacher_features1, p=2, dim=1)


            tt_dist_mat = f_t.double().mm(gallery_f.double().t())
            value, index = torch.sort(tt_dist_mat, dim=1, descending=True)
            top_k_index = index[:, :self.k]


            t_value = value[:, :self.k]
            s_value = []

            for i, s_feature in enumerate(f_s):
                s_feature = s_feature.unsqueeze(0)
                s_value.append(s_feature.double().mm(gallery_f[top_k_index[i], :].double().t()))

            s_value = torch.cat(s_value, dim=0)

        elif self.mode == 'l2':
            if layer == 0:
                gallery_f = self.teacher_features
            elif layer == 1:
                gallery_f = self.teacher_features1

            tt_dist_mat = euclidean_dist(f_t.double(), gallery_f.double())

            value, index = torch.sort(tt_dist_mat, dim=1, descending=False)
            top_k_index = index[:, :self.k]
            t_value = value[:, :self.k]
            s_value = []

            for i, s_feature in enumerate(f_s):
                s_feature = s_feature.unsqueeze(0)
                s_value.append(euclidean_dist(s_feature.double(), gallery_f[top_k_index[i], :].double()))

            s_value = torch.cat(s_value, dim=0)

        main_mat = (s_value[:, 0] - t_value[:, 0]).abs()
        main_loss = self.alpha * main_mat.mean()

        repeat_size = t_value[:, 1:].size(1)
        t_rank = smooth_rank(t_value[:, 1:], repeat_size, self.mode)
        s_rank = smooth_rank(s_value[:, 1:], repeat_size, self.mode)

        extra_loss = self.beta * (torch.norm(t_rank - s_rank, p=2, dim=1)).mean() / repeat_size

        return main_loss + extra_loss

