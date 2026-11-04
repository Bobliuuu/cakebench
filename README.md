# cakebench

Kernel benchmark modelled on **CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution** (arXiv 2608.12629).
Scores a candidate kernel (hand-written or LLM-generated) on the paper's kernel families:

| task | paper analogue |
|---|---|
| `kmeans_assign` | Flash-KMeans assign (B=32,N=65536,K=1024,D=128) |
| `knn_search` | KNN dispatcher portfolio (recall must be ~1.0) |
| `tinygemm` | small-M BF16 GEMM, 12 decode shapes |
| `kda_decode` | Kimi Delta Attention decode |
| `kda_prefill` | Kimi Delta Attention prefill from cached state |

## Method (mirrors paper)
- Candidate = file defining `kernel(**inputs)`; checked against an fp32 oracle over every shape x 3 input distributions (normal / heavy-tailed / tiny).
- Timing: median GPU span, L2 flushed before each sample (CUDA events; `--cupti` for torch.profiler CUPTI span).
- Score: per-shape speedup over a torch baseline; geometric mean over the portfolio. Incorrect anywhere => no score.
- Failures return localized diagnostics (shape, dist, metric) that the agent loop feeds back to the LLM.

## Use
```
pip install -e .[test]
cakebench list
cakebench run kmeans_assign examples/kmeans_assign.py            # small shapes on CPU, full on GPU with --full
cakebench suite --device cuda --full                               # all examples/
export OPENROUTER_API_KEY=...
cakebench agent tinygemm --model anthropic/claude-sonnet-4.5 --rounds 6 --device cuda --full
pytest
```

## Caveats
- Cake IR / compiler is not public, so this benches *kernels*, not the compiler. Baselines are torch, not the paper's FlashML/FlashKDA/FlashInfer.
- Paper numbers are B200; full portfolios need a large GPU. Only CPU path has been run here (tests pass); CUDA/CUPTI paths are untested.
- Agent loop tested with a mock chat function; no live OpenRouter call made.
