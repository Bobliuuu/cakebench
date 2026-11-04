import torch


def kernel(q, k, v, g, beta, state):
    S = state * g.exp().unsqueeze(-1)
    kf = k.float()
    u = beta.unsqueeze(-1) * (v.float() - torch.einsum("bhk,bhkv->bhv", kf, S))
    S = S + kf.unsqueeze(-1) * u.unsqueeze(-2)
    o = torch.einsum("bhk,bhkv->bhv", q.float(), S) * (q.shape[-1] ** -0.5)
    return o.to(q.dtype), S
