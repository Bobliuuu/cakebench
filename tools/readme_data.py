"""Combine the task registry (cakebench.tasks) with results/a100_example_status.json -> results/derived/readme_data.json."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cakebench.tasks import TASKS, DISTS
rec = json.load(open(os.path.join(ROOT, "results", "a100_example_status.json")))
rows = []
for r in rec["rows"]:
    t = TASKS[r["task"]]
    rows.append({**r, "task": r["task"], "shapes_full": len(t.shapes), "shapes_small": len(t.small_shapes),
                 "correct": f'{r["shapes_correct"]}/{r["shapes_total"]}', "desc": t.description})
sp = [r["geomean_speedup"] for r in rows]
import math
out = {"rows": rows, "n_tasks": len(TASKS), "n_dists": len(DISTS), "total_shapes": sum(len(t.shapes) for t in TASKS.values()),
       "max_speedup": max(sp), "all_correct": all(r["shapes_correct"] == r["shapes_total"] for r in rows),
       "suite_geomean": math.exp(sum(math.log(s) for s in sp) / len(sp)), "provenance": rec["provenance"]}
os.makedirs(os.path.join(ROOT, "results", "derived"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "derived", "readme_data.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k not in ("rows", "provenance")})
