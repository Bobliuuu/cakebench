import argparse
import json
import sys

import torch

from .agent import run_agent
from .harness import evaluate_file, format_diagnostics
from .tasks import TASKS


def main(argv=None):
    ap = argparse.ArgumentParser("cakebench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run", help="evaluate candidate file")
    r.add_argument("task", choices=list(TASKS))
    r.add_argument("candidate")
    a = sub.add_parser("agent", help="LLM agent via OpenRouter (needs OPENROUTER_API_KEY)")
    a.add_argument("task", choices=list(TASKS))
    a.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    a.add_argument("--rounds", type=int, default=5)
    a.add_argument("--out", default="runs")
    s = sub.add_parser("suite", help="evaluate examples/<task>.py for all tasks and print geomean")
    s.add_argument("--dir", default="examples")
    for p in (r, a, s):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--full", action="store_true", help="full GPU shape portfolio")
        p.add_argument("--iters", type=int, default=20)
        p.add_argument("--cupti", action="store_true")
        p.add_argument("--json", action="store_true")
    n = ap.parse_args(argv)
    if n.cmd == "list":
        for t in TASKS.values():
            print(f"{t.name:14s} {len(t.shapes):3d} shapes  {t.description}")
        return 0
    kw = dict(device=n.device, small=not n.full, iters=n.iters, cupti=n.cupti)
    if n.cmd == "run":
        res = evaluate_file(n.task, n.candidate, **kw)
        print(json.dumps(res, indent=1, default=str) if n.json else format_diagnostics(res))
        return 0 if res["all_correct"] else 1
    if n.cmd == "agent":
        print(json.dumps(run_agent(n.task, n.model, n.out, rounds=n.rounds, **kw), indent=1))
        return 0
    if n.cmd == "suite":
        from pathlib import Path
        from .timing import geomean
        sp, bad = [], 0
        for t in TASKS:
            f = Path(n.dir) / f"{t}.py"
            if not f.exists():
                continue
            res = evaluate_file(t, f, **kw)
            print(f"== {t}\n{format_diagnostics(res)}")
            if res["all_correct"]:
                sp.append(res["geomean_speedup"])
            else:
                bad += 1
        print(f"SUITE geomean speedup {geomean(sp):.3f} over {len(sp)} tasks, {bad} failing")
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
