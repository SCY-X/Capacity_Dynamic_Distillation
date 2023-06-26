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


class XBM(nn.Module):
    def __init__(self, size, embed_dim):
        super(XBM, self).__init__()
        self.size = size

        # Create memory bank, modified from
        # https://github.com/facebookresearch/moco/blob/master/moco/builder.py
        self.register_buffer("embed_queue", torch.randn(size, embed_dim))
        # self.embed_queue = F.normalize(self.embed_queue, dim=0)
        # self.register_buffer("label_queue", torch.zeros(size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.is_full = False

    @torch.no_grad()
    def update(self, embeddings):  # , labels):
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)

        assert self.size % batch_size == 0

        # Enqueue
        self.embed_queue[ptr:ptr + batch_size, :] = embeddings
        # self.label_queue[ptr:ptr + batch_size] = labels
        if ptr + batch_size >= self.size:
            self.is_full = True
        ptr = (ptr + batch_size) % self.size

        self.queue_ptr[0] = ptr

    def get(self):
        if self.is_full:
            return self.embed_queue.cuda()  # , self.label_queue
        else:
            return self.embed_queue[:self.queue_ptr].cuda()  # , self.label_queue[:self.queue_ptr]


class Context_RGR(nn.Module):
    def __init__(self, size, queue_num, embed_dim, ratio, k, dist):
        super(Context_RGR, self).__init__()
        self.t_sum_queue = []
        self.ratio = ratio
        self.k = k
        self.size = size
        self.embed_dim = embed_dim
        self.dist = dist

        for i in range(queue_num):
            exec('self.t_queue%s = XBM(self.size, self.embed_dim[%s])' % (i, i))
            exec('self.t_sum_queue.append(self.t_queue%s)' % i)

    def forward(self, model, s_map, t_map):
        compactor_name_to_mask = {}
        compactor_name_to_kernel_param = {}

        for name, buffer in model.named_buffers():
            if 'compactor_mask' in name:
                compactor_name_to_mask[name.replace('_mask', '')] = buffer

        for name, param in model.named_parameters():
            if 'compactor' in name:
                compactor_name_to_kernel_param[name.replace('.weight', '')] = param

        mask_index = []
        for i, s_f in enumerate(s_map):
            length = s_f.size(1)
            t_f = t_map[i]

            self.t_sum_queue[i].update(t_f)

            t_gallery = self.t_sum_queue[i].get()

            if self.dist == "cosine":
                t_f = F.normalize(t_f, p=2, dim=1)
                s_f = F.normalize(s_f, p=2, dim=1)
                t_gallery = F.normalize(t_gallery, p=2, dim=1)

                retri_result = t_f.double().mm(t_gallery.double().t())

                value, retri_index = torch.sort(retri_result, dim=1, descending=True)

                sum_index = retri_index[:, :self.k]
                s_index = torch.arange(0, length).cuda()

                for k in range(0, self.k):
                    simiar_t_f = t_gallery[sum_index[:, k], :]
                    distance_martix = (simiar_t_f * s_f).abs()

                    _, index = torch.sort(distance_martix, descending=False)
                    index = index[:, :int(length * self.ratio)]

                    for j in range(index.size(0)):
                        a_cat_b, counts = torch.cat([s_index, index[j]]).unique(return_counts=True)
                        s_index = a_cat_b[torch.where(counts.gt(1))]

                mask_index.append(s_index)


            elif self.dist == "l2":
                retri_result = euclidean_dist(t_f.double(), t_gallery.double())
                value, retri_index = torch.sort(retri_result, dim=1, descending=False)

                sum_index = retri_index[:, :self.k]
                s_index = torch.arange(0, length).cuda()

                for k in range(0, self.k):
                    simiar_t_f = t_gallery[sum_index[:, k], :]
                    distance_martix = (simiar_t_f * s_f).abs()

                    _, index = torch.sort(distance_martix, descending=False)
                    index = index[:, :int(length * self.ratio)]

                    for j in range(index.size(0)):
                        a_cat_b, counts = torch.cat([s_index, index[j]]).unique(return_counts=True)
                        s_index = a_cat_b[torch.where(counts.gt(1))]

                mask_index.append(s_index)

        result = {}
        for i, (name, kernel) in enumerate(compactor_name_to_kernel_param.items()):

            mask = compactor_name_to_mask[name]
            mask_zero_index = mask_index[i]

            if len(mask_zero_index) > 0:
                mask[mask_zero_index] = 0
            num_filters = mask.nelement()

            if kernel.ndimension() == 4:
                if mask.ndimension() == 1:
                    broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
                    result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
                else:
                    assert mask.ndimension() == 4
                    result[kernel] = mask
            else:
                assert kernel.ndimension() == 1
                result[kernel] = mask
        return result
