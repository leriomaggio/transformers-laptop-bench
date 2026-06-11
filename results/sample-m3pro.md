## Results: HuggingFaceTB/SmolLM2-1.7B-Instruct

Device `mps` on Apple M3 Pro, 36.0 GB RAM. 10 measured runs, 2 warmup, max_new_tokens=128, seed 0.

torch 2.12.0, transformers 5.11.0, optimum-quanto 0.2.7, Python 3.13.13.

| Config | Prompt | In tok | Out tok | Load s | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Tok/s p50 | Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | short | 63 | 128 | 1.76 | 0.063 | 0.070 | 4.532 | 4.593 | 28.2 | 3302 |
| bf16 | long | 271 | 128 | 1.76 | 0.236 | 0.253 | 5.061 | 5.503 | 25.3 | 3355 |
| int8 | short | 63 | 128 | 1.82 | 0.237 | 0.269 | 27.808 | 28.840 | 4.6 | 3594 |
| int8 | long | 271 | 128 | 1.82 | 0.452 | 0.466 | 30.253 | 31.533 | 4.2 | 4479 |

Peak memory counted via torch.mps.current_allocated_memory (peak sampled).

