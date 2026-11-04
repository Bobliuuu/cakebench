import torch


def kernel(q, k, v, g, beta, state):
    S = state.clone()
    qf, kf, vf = q.float(), k.float(), v.float()
    decay = g.exp()
    scale = q.shape[-1] ** -0.5
    out = torch.empty(vf.shape, dtype=torch.float32, device=q.device)
    for t in range(q.shape[1]):
        S = S * decay[:, t].unsqueeze(-1)
        u = beta[:, t].unsqueeze(-1) * (vf[:, t] - torch.einsum("bhk,bhkv->bhv", kf[:, t], S))
        S = S + kf[:, t].unsqueeze(-1) * u.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", qf[:, t], S) * scale
    return out.to(q.dtype), S
