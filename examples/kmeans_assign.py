import torch


def kernel(x, c):
    # ||x||^2 is constant per point, so drop it from the argmin.
    d = (c.float() * c.float()).sum(-1)[:, None, :] * 0.5 - torch.bmm(x, c.transpose(1, 2)).float()
    return d.argmin(-1).to(torch.int32)
