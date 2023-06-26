import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_aff(x, y):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, y.t())

class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert(batch_size%num_id==0)
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, f_s, f_t):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        # ------ differentiable ranking of all retrieval set ------
        ## compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(f_s, f_t)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)

        # pass through the sigmoid
        sim_sg = F.relu(sim_diff, inplace=True) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = f_s.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        xt = f_t.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)

        # compute the relevance scores
        sim_pos = torch.bmm(xs, xt.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = F.relu(sim_pos_diff, inplace=True) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)


class APKD(nn.Module):
    """smooth AP"""
    def __init__(self, batch_size, num_id, feat_dims):
        super(APKD, self).__init__()
        self.apkd = SmoothAP(batch_size, num_id, feat_dims)

    def forward(self, outputs, targets):
        s_f = F.normalize(F.adaptive_avg_pool2d(outputs, 1).squeeze(), p=2)
        t_f = F.normalize(F.adaptive_avg_pool2d(targets, 1).squeeze(), p=2)
        loss = self.apkd(s_f, t_f) + self.apkd(t_f, s_f)
        return loss

class APKDA(nn.Module):
    """smooth AP"""
    def __init__(self, batch_size, num_id, feat_dims):
        super(APKDA, self).__init__()
        self.apkd = SmoothAP(batch_size * 2, num_id, feat_dims)
        self.num_id = num_id
        self.batch_size = batch_size
        self.feat_dims = feat_dims
        self.per_num = self.batch_size // self.num_id

    def forward(self, outputs, targets):
        s_f = F.normalize(F.adaptive_avg_pool2d(outputs, 1).squeeze(), p=2)
        t_f = F.normalize(F.adaptive_avg_pool2d(targets, 1).squeeze(), p=2)
        f = torch.zeros((self.batch_size * 2, self.feat_dims)).cuda()
        for i in range(0, self.num_id):
            f[2*i * self.per_num:(2*i+1) * self.per_num] = s_f[i * self.per_num:(i+1) * self.per_num]
            f[(2*i + 1) * self.per_num:(2*i + 2) * self.per_num] = t_f[i * self.per_num:(i+1) * self.per_num]

        loss = self.apkd(f, f)
        return loss
    
