"""Task registry. Each task = spec + input generator + fp32 oracle + correctness check + baseline.

A candidate is a python file defining ``kernel(**inputs)`` returning a tensor (or tuple of tensors)
with the same structure as the task oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

DISTS = ("normal", "heavy", "small")


def _rand(shape, dist, gen, device, dtype):
    x = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
    if dist == "heavy":  # outliers
        x = x * torch.where(torch.rand(shape, generator=gen) < 0.01, 20.0, 1.0)
    elif dist == "small":
        x = x * 1e-2
    return x.to(device=device, dtype=dtype)


@dataclass
class Task:
    name: str
    description: str
    input_names: list[str]
    shapes: dict[str, dict]          # full (GPU) portfolio
    small_shapes: dict[str, dict]    # CPU-friendly portfolio
    make_inputs: Callable            # (params, dist, gen, device) -> dict
    oracle: Callable                 # (**inputs) -> fp32 reference output
    check: Callable                  # (out, inputs, params) -> (ok, metrics dict)
    baseline: Callable               # (**inputs) -> output, "tuned reference" timing target
    spec: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- kmeans assign
def _km_inputs(p, dist, gen, device):
    dt = torch.bfloat16 if device != "cpu" else torch.float32
    return {"x": _rand((p["B"], p["N"], p["D"]), dist, gen, device, dt),
            "c": _rand((p["B"], p["K"], p["D"]), dist, gen, device, dt)}


def _km_dist(x, c):
    x, c = x.float(), c.float()
    return (x * x).sum(-1, keepdim=True) - 2 * x @ c.transpose(1, 2) + (c * c).sum(-1)[:, None, :]


def _km_oracle(x, c):
    return _km_dist(x, c).argmin(-1).to(torch.int32)


def _km_check(out, inp, p):
    d = _km_dist(inp["x"], inp["c"])
    if tuple(out.shape) != tuple(d.shape[:2]):
        return False, {"error": f"shape {tuple(out.shape)} != {tuple(d.shape[:2])}"}
    idx = out.long()
    if idx.min() < 0 or idx.max() >= d.shape[-1]:
        return False, {"error": "index out of range"}
    chosen = d.gather(-1, idx[..., None])[..., 0]
    best = d.min(-1).values
    # bf16 matmul error scales with ||x||^2+||c||^2 (not with the distance), so normalise by that
    xx = (inp["x"].float() ** 2).sum(-1)
    cc = (inp["c"].float() ** 2).sum(-1).gather(-1, idx)
    gap = ((chosen - best) / (xx + cc + 1e-6)).max().item()
    return gap < 1e-2, {"max_rel_dist_gap": gap, "exact_frac": (idx == d.argmin(-1)).float().mean().item()}


def _km_base(x, c):
    return _km_dist(x, c).argmin(-1).to(torch.int32)


# ---------------------------------------------------------------- knn search
def _knn_inputs(p, dist, gen, device):
    dt = torch.float16 if device != "cpu" else torch.float32
    return {"q": _rand((p["Q"], p["D"]), dist, gen, device, dt),
            "db": _rand((p["N"], p["D"]), dist, gen, device, dt), "k": p["k"]}


def _knn_d(q, db):
    q, db = q.float(), db.float()
    return (q * q).sum(-1, keepdim=True) - 2 * q @ db.T + (db * db).sum(-1)[None]


def _knn_oracle(q, db, k):
    return _knn_d(q, db).topk(k, dim=-1, largest=False).indices.to(torch.int32)


def _knn_check(out, inp, p):
    d = _knn_d(inp["q"], inp["db"])
    k = inp["k"]
    if tuple(out.shape) != (d.shape[0], k):
        return False, {"error": f"shape {tuple(out.shape)} != {(d.shape[0], k)}"}
    idx = out.long()
    if idx.min() < 0 or idx.max() >= d.shape[1]:
        return False, {"error": "index out of range"}
    kth = d.topk(k, dim=-1, largest=False).values[:, -1:]
    dd = d.gather(-1, idx)
    qq = (inp["q"].float() ** 2).sum(-1, keepdim=True)
    scale = qq + (inp["db"].float() ** 2).sum(-1)[idx]  # fp16 matmul error scales with the norms
    hit = (dd <= kth + 5e-3 * scale).float()
    srt = idx.sort(-1).values
    uniq = bool((srt[:, 1:] != srt[:, :-1]).all().item()) if k > 1 else True
    recall = hit.mean().item()
    return bool(uniq and recall >= 0.999), {"recall": recall, "unique": bool(uniq)}


def _knn_base(q, db, k):
    return _knn_d(q, db).topk(k, dim=-1, largest=False).indices.to(torch.int32)


# ---------------------------------------------------------------- tinygemm
def _gm_inputs(p, dist, gen, device):
    dt = torch.bfloat16 if device != "cpu" else torch.float32
    return {"x": _rand((p["M"], p["K"]), dist, gen, device, dt),
            "w": _rand((p["N"], p["K"]), dist, gen, device, dt)}


def _gm_oracle(x, w):
    return x.float() @ w.float().T


def _gm_check(out, inp, p):
    ref = _gm_oracle(inp["x"], inp["w"])
    if tuple(out.shape) != tuple(ref.shape):
        return False, {"error": f"shape {tuple(out.shape)} != {tuple(ref.shape)}"}
    err = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
    return err < 2e-2 and bool(torch.isfinite(out.float()).all()), {"max_rel_err": err}


def _gm_base(x, w):
    return x @ w.T


# ---------------------------------------------------------------- KDA (delta-rule) decode + prefill
def _kda_inputs_t(p, dist, gen, device, T):
    B, H, Dk, Dv = p["B"], p["H"], p["Dk"], p["Dv"]
    dt = torch.bfloat16 if device != "cpu" else torch.float32
    sh = (B, T, H) if T else (B, H)
    q = _rand(sh + (Dk,), dist, gen, device, dt)
    k = torch.nn.functional.normalize(_rand(sh + (Dk,), dist, gen, device, torch.float32), dim=-1).to(dt)
    v = _rand(sh + (Dv,), dist, gen, device, dt)
    g = (-torch.rand(sh + (Dk,), generator=gen) * 0.1).to(device=device, dtype=torch.float32)  # log-decay
    beta = torch.rand(sh, generator=gen).to(device=device, dtype=torch.float32)
    s0 = (_rand((B, H, Dk, Dv), dist, gen, device, torch.float32) * 0.1)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "state": s0}


def _kda_step(S, q, k, v, g, beta):
    S = S * g.exp()[..., None]
    u = beta[..., None] * (v - torch.einsum("bhk,bhkv->bhv", k, S))
    S = S + k[..., None] * u[..., None, :]
    o = torch.einsum("bhk,bhkv->bhv", q, S) * (q.shape[-1] ** -0.5)
    return o, S


def _kda_decode_inputs(p, dist, gen, device):
    return _kda_inputs_t(p, dist, gen, device, 0)


def _kda_decode_ref(q, k, v, g, beta, state, dtype=torch.float32):
    o, S = _kda_step(state.to(dtype), q.to(dtype), k.to(dtype), v.to(dtype), g.to(dtype), beta.to(dtype))
    return o, S


def _kda_prefill_inputs(p, dist, gen, device):
    return _kda_inputs_t(p, dist, gen, device, p["T"])


def _kda_prefill_ref(q, k, v, g, beta, state, dtype=torch.float32):
    S = state.to(dtype)
    outs = []
    for t in range(q.shape[1]):
        o, S = _kda_step(S, q[:, t].to(dtype), k[:, t].to(dtype), v[:, t].to(dtype),
                         g[:, t].to(dtype), beta[:, t].to(dtype))
        outs.append(o)
    return torch.stack(outs, 1), S


def _kda_check(out, inp, p):
    ref_fn = _kda_prefill_ref if "T" in p else _kda_decode_ref
    ro, rs = ref_fn(**inp)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        return False, {"error": "expected (output, new_state)"}
    o, s = out
    if tuple(o.shape) != tuple(ro.shape) or tuple(s.shape) != tuple(rs.shape):
        return False, {"error": f"shapes {tuple(o.shape)},{tuple(s.shape)} != {tuple(ro.shape)},{tuple(rs.shape)}"}
    eo = ((o.float() - ro).abs().max() / (ro.abs().max() + 1e-6)).item()
    es = ((s.float() - rs).abs().max() / (rs.abs().max() + 1e-6)).item()
    ok = eo < 3e-2 and es < 3e-2 and bool(torch.isfinite(o.float()).all())
    return ok, {"out_rel_err": eo, "state_rel_err": es}


def _kda_decode_base(q, k, v, g, beta, state):
    o, S = _kda_decode_ref(q, k, v, g, beta, state, dtype=torch.float32)
    return o.to(q.dtype), S


def _kda_prefill_base(q, k, v, g, beta, state):
    o, S = _kda_prefill_ref(q, k, v, g, beta, state, dtype=torch.float32)
    return o.to(q.dtype), S


def _id(**kw):  # oracle wrappers keep keyword form
    raise NotImplementedError


TASKS: dict[str, Task] = {}


def _reg(t: Task):
    TASKS[t.name] = t


_reg(Task(
    "kmeans_assign", "K-Means assignment: nearest centroid (L2) per point. Paper: Flash-KMeans assign, B200.",
    ["x", "c"],
    {"paper": dict(B=32, N=65536, K=1024, D=128), "mid": dict(B=8, N=32768, K=512, D=128),
     "wide": dict(B=16, N=16384, K=2048, D=64)},
    {"s0": dict(B=2, N=256, K=16, D=32), "s1": dict(B=1, N=513, K=33, D=64)},
    _km_inputs, _km_oracle, _km_check, _km_base,
    "kernel(x[B,N,D], c[B,K,D]) -> int32[B,N] index of nearest centroid (squared L2)."))
_reg(Task(
    "knn_search", "Exact k-NN search; recall must be ~1.0 and indices unique. Paper: dispatcher portfolio.",
    ["q", "db", "k"],
    {"a": dict(Q=1024, N=262144, D=128, k=16), "b": dict(Q=4096, N=65536, D=96, k=32),
     "c": dict(Q=256, N=1048576, D=64, k=8)},
    {"s0": dict(Q=16, N=300, D=32, k=5), "s1": dict(Q=7, N=129, D=16, k=1)},
    _knn_inputs, _knn_oracle, _knn_check, _knn_base,
    "kernel(q[Q,D], db[N,D], k:int) -> int32[Q,k] indices of k nearest db rows per query (squared L2)."))
_reg(Task(
    "tinygemm", "Small-M GEMM  y = x @ w.T. Paper: TinyGEMM, 35 decode shapes.",
    ["x", "w"],
    {f"m{m}_n{n}": dict(M=m, N=n, K=k) for m in (1, 4, 16, 32) for n, k in ((4096, 4096), (11008, 4096), (4096, 11008))},
    {"s0": dict(M=1, N=64, K=96), "s1": dict(M=8, N=100, K=48)},
    _gm_inputs, _gm_oracle, _gm_check, _gm_base,
    "kernel(x[M,K], w[N,K]) -> [M,N] same dtype as x."))
_reg(Task(
    "kda_decode", "Kimi Delta Attention single-step decode with per-channel decay (gated delta rule).",
    ["q", "k", "v", "g", "beta", "state"],
    {f"b{b}": dict(B=b, H=32, Dk=128, Dv=128) for b in (1, 8, 32, 128)},
    {"s0": dict(B=2, H=3, Dk=16, Dv=16), "s1": dict(B=1, H=2, Dk=8, Dv=24)},
    _kda_decode_inputs, lambda **kw: _kda_decode_ref(**kw), _kda_check, _kda_decode_base,
    "kernel(q,k[B,H,Dk], v[B,H,Dv], g[B,H,Dk] log-decay fp32, beta[B,H] fp32, state[B,H,Dk,Dv] fp32)"
    " -> (o[B,H,Dv], new_state fp32). S=S*exp(g); u=beta*(v-k^T S); S+=k u^T; o=(q^T S)/sqrt(Dk)."))
_reg(Task(
    "kda_prefill", "Kimi Delta Attention prefill over T tokens from cached state (sequential delta-rule oracle).",
    ["q", "k", "v", "g", "beta", "state"],
    {f"t{t}": dict(B=b, T=t, H=16, Dk=128, Dv=128) for b, t in ((1, 2048), (4, 1024), (16, 256))},
    {"s0": dict(B=2, T=9, H=2, Dk=8, Dv=8), "s1": dict(B=1, T=17, H=1, Dk=16, Dv=8)},
    _kda_prefill_inputs, lambda **kw: _kda_prefill_ref(**kw), _kda_check, _kda_prefill_base,
    "kernel(q,k[B,T,H,Dk], v[B,T,H,Dv], g[B,T,H,Dk], beta[B,T,H], state[B,H,Dk,Dv]) -> (o[B,T,H,Dv], final_state)."
    " Same recurrence as kda_decode applied for t=0..T-1."))
