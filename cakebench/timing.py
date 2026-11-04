"""Median GPU span timing with L2 flush before each sample (CUDA events; CUPTI via torch.profiler optional)."""
from __future__ import annotations

import statistics
import time

import torch

_flush_buf = None


def _flush(device):
    global _flush_buf
    if _flush_buf is None or _flush_buf.device != torch.device(device):
        _flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)  # > B200 L2
    _flush_buf.zero_()


def time_fn(fn, device="cpu", iters=30, warmup=5, flush=True, cupti=False):
    """Return (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    ts = []
    if is_cuda and cupti:
        from torch.profiler import ProfilerActivity, profile
        for _ in range(iters):
            if flush:
                _flush(device)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            if evs:  # GPU span: first start -> last end
                ts.append((max(e.time_range.end for e in evs) - min(e.time_range.start for e in evs)) / 1e3)
    elif is_cuda:
        for _ in range(iters):
            if flush:
                _flush(device)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    else:
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
    if not ts:
        raise RuntimeError("no timing samples")
    return statistics.median(ts), min(ts), max(ts)


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
