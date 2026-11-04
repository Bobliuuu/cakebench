import torch.nn.functional as F


def kernel(x, w):
    return F.linear(x, w)
