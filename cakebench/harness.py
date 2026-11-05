"""Evaluate a candidate kernel file on a task: load, correctness (shapes x distributions), timing, speedup."""
from __future__ import annotations

import importlib.util
import json
import traceback
import hashlib
from pathlib import Path

import torch

from .tasks import DISTS, TASKS
from .timing import geomean, time_fn


def load_candidate(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("cake_cand_" + hashlib.md5(str(path).encode()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel"):
        raise AttributeError("candidate must define `kernel(**inputs)`")
    return mod.kernel


def _call(fn, inputs):
    return fn(**inputs)


def _describe(e):
    """Exception text incl. the cause chain (e.g. Triton CompilationError hides the reason in __cause__)."""
    parts, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    return " <- caused by: ".join(parts)


def _clone(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}


def evaluate(task_name, fn, device="cpu", small=None, dists=DISTS, iters=20, cupti=False,
             seed=0, baseline_fn=None, shapes_filter=None):
    """Returns a result dict. Diagnostics are localized: per shape/dist failure with metrics."""
    task = TASKS[task_name]
    small = (device == "cpu") if small is None else small
    portfolio = task.small_shapes if small else task.shapes
    if shapes_filter:
        portfolio = {k: v for k, v in portfolio.items() if k in shapes_filter}
    baseline_fn = baseline_fn or task.baseline
    rows, all_ok = [], True
    for sname, p in portfolio.items():
        row = {"shape": sname, "params": p, "correct": True, "failures": []}
        gen = torch.Generator().manual_seed(seed)
        for dist in dists:
            inputs = task.make_inputs(p, dist, gen, device)
            try:
                out = _call(fn, _clone(inputs))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                ok, m = task.check(out, inputs, p)
            except Exception as e:  # compile/runtime errors go back to the agent
                ok, m = False, {"error": _describe(e), "trace": traceback.format_exc()[-800:]}
            if not ok:
                row["correct"] = False
                row["failures"].append({"dist": dist, **m})
        if row["correct"]:
            inputs = task.make_inputs(p, "normal", torch.Generator().manual_seed(seed), device)
            try:
                c = time_fn(lambda: _call(fn, inputs), device, iters, cupti=cupti)
                b = time_fn(lambda: _call(baseline_fn, inputs), device, iters, cupti=cupti)
                row.update(cand_ms=c[0], base_ms=b[0], speedup=b[0] / c[0])
            except Exception as e:
                row["correct"] = False
                row["failures"].append({"dist": "normal", "error": f"timing: {type(e).__name__}: {e}"})
        all_ok &= row["correct"]
        rows.append(row)
    sp = [r["speedup"] for r in rows if r.get("speedup")]
    return {"task": task_name, "device": str(device), "all_correct": all_ok,
            "geomean_speedup": geomean(sp) if all_ok else None,
            "n_correct": sum(r["correct"] for r in rows), "n_shapes": len(rows), "rows": rows}


def evaluate_file(task_name, path, **kw):
    try:
        fn = load_candidate(path)
    except Exception as e:
        return {"task": task_name, "all_correct": False, "geomean_speedup": None, "rows": [],
                "n_correct": 0, "n_shapes": 0, "load_error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}
    return evaluate(task_name, fn, **kw)


def _short(f):
    """Failure -> compact text. For errors keep head AND tail (the real message is at the end)."""
    d = {k: v for k, v in f.items() if k not in ("dist", "trace")}
    e = d.get("error")
    if isinstance(e, str) and len(e) > 900:
        d["error"] = e[:300] + " ... " + e[-600:]
    return json.dumps(d)


def format_diagnostics(res, limit=6):
    """Compact localized feedback for the agent. Identical failures across distributions are merged."""
    if res.get("load_error"):
        return "LOAD ERROR:\n" + res["load_error"][-1500:]
    lines = [f"correct {res['n_correct']}/{res['n_shapes']} shapes; geomean speedup vs baseline: {res['geomean_speedup']}"]
    n = 0
    for r in res["rows"]:
        if r["correct"]:
            lines.append(f"  OK   {r['shape']}: {r['cand_ms']:.4f} ms (baseline {r['base_ms']:.4f}) x{r['speedup']:.2f}")
            continue
        merged = {}
        for f in r["failures"]:
            merged.setdefault(_short(f) if "error" in f else None, []).append(f)
        for key, fs in merged.items():
            if n >= limit:
                break
            dists = ",".join(f.get("dist", "?") for f in fs)
            body = key if key else "; ".join(f"{f['dist']}: " + _short(f) for f in fs)
            lines.append(f"  FAIL {r['shape']} {r['params']} dist={dists}: {body}")
            n += 1
    return "\n".join(lines)
