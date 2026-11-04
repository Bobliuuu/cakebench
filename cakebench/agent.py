"""LLM kernel-writing agent over OpenRouter. Evaluate -> localized diagnostics -> retry, with a token budget."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from .harness import evaluate_file, format_diagnostics
from .tasks import TASKS

URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = ("You write fast GPU kernels in Python (PyTorch, Triton, or CUDA via torch.utils.cpp_extension). "
          "Reply with ONE ```python code block defining `kernel(**inputs)` exactly per the spec. "
          "Correctness is checked against an fp32 oracle across shapes and input distributions "
          "(normal, heavy-tailed outliers, tiny magnitude); speed is median GPU span with L2 flushed, "
          "scored as geomean speedup over a torch baseline. Inputs are bf16/fp16: do all arithmetic and "
          "accumulation in fp32 (cast after loading); bf16 intermediates fail the accuracy check. "
          "Keep Triton simple (tl.dot for matmul shapes, fp32 accumulators) and make it compile for every "
          "shape, including tiny ones.")


def chat(model, messages, api_key=None, timeout=300, max_tokens=6000):
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    req = urllib.request.Request(URL, data=json.dumps({"model": model, "messages": messages,
                                                       "max_tokens": max_tokens}).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {}).get("total_tokens", 0)


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else None


def run_agent(task_name, model, out_dir, rounds=5, token_budget=200_000, chat_fn=chat, **eval_kw):
    task = TASKS[task_name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task `{task.name}`: {task.description}\nSpec: {task.spec}\n"
                                        f"Inputs: {task.input_names}. Shapes: {json.dumps(task.shapes)}"}]
    best, used, hist = None, 0, []
    for i in range(rounds):
        try:
            text, toks = chat_fn(model, msgs)
        except Exception as e:  # API/credit errors: keep partial results instead of crashing
            hist.append({"round": i, "error": f"{type(e).__name__}: {e}"})
            break
        used += toks
        code = extract_code(text)
        if code is None:
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "No code block found."}]
            hist.append({"round": i, "error": "no code"})
            continue
        path = out / f"{task_name}_r{i}.py"
        path.write_text(code)
        res = evaluate_file(task_name, path, **eval_kw)
        diag = format_diagnostics(res)
        hist.append({"round": i, "file": str(path), "tokens": toks, "correct": res["all_correct"],
                     "geomean_speedup": res["geomean_speedup"]})
        if res["all_correct"] and (best is None or res["geomean_speedup"] > best["geomean_speedup"]):
            best = {"file": str(path), "geomean_speedup": res["geomean_speedup"]}
        if used >= token_budget:
            break
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Evaluation:\n{diag}\nImprove (fix failures first, then speed)."}]
    summary = {"task": task_name, "model": model, "tokens": used, "best": best, "history": hist}
    (out / f"{task_name}_agent.json").write_text(json.dumps(summary, indent=2))
    return summary
