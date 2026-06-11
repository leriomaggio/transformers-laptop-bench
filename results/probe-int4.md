## Results: HuggingFaceTB/SmolLM2-1.7B-Instruct

Device `mps` on Apple M3 Pro, 36.0 GB RAM. 3 measured runs, 1 warmup, max_new_tokens=64, seed 0.

torch 2.12.0, transformers 5.11.0, optimum-quanto 0.2.7, Python 3.13.13.

| Config | Prompt | In tok | Out tok | Load s | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Tok/s p50 | Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int4 | short | 63 | 64 | 2.80 | 0.893 | 1.483 | 58.623 | 61.230 | 1.1 | 3706 |
| int4 | long | 271 | 64 | 2.80 | 1.007 | 1.019 | 54.797 | 55.971 | 1.2 | 4048 |

Peak memory counted via torch.mps.current_allocated_memory (peak sampled).

