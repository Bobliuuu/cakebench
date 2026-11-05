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

## Status (A100-SXM4-40GB, shared box, `flock ~/.gpu.lock`)
| example kernel (hand-written torch) | correct | geomean vs torch baseline |
|---|---|---|
| kmeans_assign | 3/3 | 1.69x |
| knn_search | 3/3 | 1.13x |
| tinygemm | 8/8 | 1.00x |
| kda_decode | 4/4 | 1.03x |
| kda_prefill | 3/3 | 1.06x |

LLM agent via OpenRouter: Haiku 4.5 produced a correct, 1.16x `tinygemm` kernel on CPU in 3 rounds.
On GPU `kmeans_assign` (Sonnet 4.5, Haiku 4.5) no model produced a correct Triton kernel yet: failures were Triton
compile errors and one bf16-accumulation accuracy failure. Diagnostics now carry the exception cause chain, merged
failures and error tails; the last runs were cut off by account credit (HTTP 402) before the fixed feedback could be exercised.

## Caveats
- Cake IR / compiler is not public, so this benches *kernels*, not the compiler. Baselines are torch, not the paper's FlashML/FlashKDA/FlashInfer.
- Paper numbers are B200; only an A100 (shared, so timings are noisy) was used. CUPTI path untested.
- No LLM-written GPU kernel has passed yet; the example speedups are from hand-written torch.
