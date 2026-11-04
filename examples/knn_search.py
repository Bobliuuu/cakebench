import torch


def kernel(q, db, k):
    d = (db.float() ** 2).sum(-1)[None] * 0.5 - (q @ db.T).float()
    return d.topk(k, dim=-1, largest=False).indices.to(torch.int32)
