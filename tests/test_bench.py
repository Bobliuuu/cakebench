import pytest
import torch

from cakebench.agent import extract_code, run_agent
from cakebench.cli import main
from cakebench.harness import evaluate, evaluate_file
from cakebench.tasks import TASKS


@pytest.mark.parametrize("name", list(TASKS))
def test_example_correct(name):
    r = evaluate_file(name, f"examples/{name}.py", iters=3)
    assert r["all_correct"], r
    assert r["geomean_speedup"] > 0


def test_oracle_as_candidate_passes():
    for name, t in TASKS.items():
        p = next(iter(t.small_shapes.values()))
        inp = t.make_inputs(p, "normal", torch.Generator().manual_seed(0), "cpu")
        ok, m = t.check(t.oracle(**inp), inp, p)
        assert ok, (name, m)


def test_wrong_kernels_rejected():
    bad = {"kmeans_assign": lambda x, c: torch.zeros(x.shape[:2], dtype=torch.int32),
           "tinygemm": lambda x, w: x @ w.T * 1.1,
           "knn_search": lambda q, db, k: torch.zeros(q.shape[0], k, dtype=torch.int32),
           "kda_decode": lambda q, k, v, g, beta, state: (torch.zeros_like(v), state),
           "kda_prefill": lambda q, k, v, g, beta, state: (torch.zeros_like(v), state)}
    for name, fn in bad.items():
        r = evaluate(name, fn, iters=2)
        assert not r["all_correct"] and r["geomean_speedup"] is None, name


def test_exception_reported_not_raised():
    r = evaluate("tinygemm", lambda x, w: 1 / 0, iters=2)
    assert not r["all_correct"] and "ZeroDivisionError" in r["rows"][0]["failures"][0]["error"]


def test_agent_loop_with_mock(tmp_path):
    bad = "```python\ndef kernel(x, w):\n    return x\n```"
    good = "```python\nimport torch\ndef kernel(x, w):\n    return x @ w.T\n```"
    replies = iter([(bad, 100), (good, 100)])
    seen = []

    def fake(model, msgs):
        seen.append(msgs[-1]["content"])
        return next(replies)

    s = run_agent("tinygemm", "m", tmp_path, rounds=2, chat_fn=fake, iters=2)
    assert s["best"] and s["tokens"] == 200
    assert "FAIL" in seen[1]  # localized diagnostics were fed back


def test_extract_code():
    assert extract_code("x ```python\na=1\n``` y") == "a=1\n"
    assert extract_code("none") is None


def test_cli(capsys):
    assert main(["list"]) == 0
    assert main(["suite", "--device", "cpu", "--iters", "2"]) == 0
    assert "SUITE geomean" in capsys.readouterr().out


def test_diagnostics_merge_and_tail():
    from cakebench.harness import format_diagnostics
    err = "X" * 2000 + "REAL_MESSAGE"
    res = {"n_correct": 0, "n_shapes": 1, "geomean_speedup": None, "rows": [{
        "shape": "s", "params": {}, "correct": False,
        "failures": [{"dist": d, "error": err} for d in ("normal", "heavy", "small")]}]}
    out = format_diagnostics(res)
    assert out.count("FAIL") == 1 and "normal,heavy,small" in out and "REAL_MESSAGE" in out
