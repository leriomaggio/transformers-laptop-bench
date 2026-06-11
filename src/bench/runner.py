"""Sweep orchestration: configs x prompt lengths, with clean teardown.

A fresh model is loaded for each config so memory is measured from a clean
state and quantisation does not carry over. Load time is recorded per config
and kept out of the inference timings.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import torch

from . import model as model_mod
from .measure import RunMetrics, count_input_tokens, measure_config

log = logging.getLogger("bench.runner")


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


@dataclass
class CellResult:
    config: str
    prompt_label: str
    input_tokens: int
    output_tokens: int
    load_seconds: float
    ttft_p50_s: float
    ttft_p95_s: float
    total_p50_s: float
    total_p95_s: float
    throughput_p50_tok_s: float
    throughput_p95_tok_s: float
    peak_memory_mb: float
    memory_method: str = ""
    skipped: bool = False
    skip_reason: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SweepResult:
    model_id: str
    device: str
    max_new_tokens: int
    warmup: int
    runs: int
    seed: int
    cells: list[CellResult] = field(default_factory=list)


def _seed_everything(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


def _aggregate(
    config: str,
    label: str,
    input_tokens: int,
    load_seconds: float,
    metrics: list[RunMetrics],
    peak_bytes: int,
    memory_method: str,
) -> CellResult:
    ttft = [m.ttft_s for m in metrics]
    total = [m.total_s for m in metrics]
    thru = [m.throughput_tok_s for m in metrics]
    return CellResult(
        config=config,
        prompt_label=label,
        input_tokens=input_tokens,
        output_tokens=metrics[0].output_tokens,
        load_seconds=round(load_seconds, 3),
        ttft_p50_s=round(_percentile(ttft, 50), 4),
        ttft_p95_s=round(_percentile(ttft, 95), 4),
        total_p50_s=round(_percentile(total, 50), 4),
        total_p95_s=round(_percentile(total, 95), 4),
        throughput_p50_tok_s=round(_percentile(thru, 50), 2),
        throughput_p95_tok_s=round(_percentile(thru, 95), 2),
        peak_memory_mb=round(peak_bytes / 1024**2, 1),
        memory_method=memory_method,
    )


def _free(loaded, device: str) -> None:
    if loaded.model is not None:
        del loaded.model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def run_sweep(
    model_id: str,
    configs: list[str],
    device: str,
    prompts: dict[str, str],
    max_new_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
) -> SweepResult:
    result = SweepResult(
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        runs=runs,
        seed=seed,
    )

    for config in configs:
        log.info("loading config '%s'", config)
        _seed_everything(seed, device)
        loaded = model_mod.load(model_id, config, device)

        if loaded.skipped:
            log.warning("skipping '%s': %s", config, loaded.skip_reason)
            for label in prompts:
                result.cells.append(
                    CellResult(
                        config=config,
                        prompt_label=label,
                        input_tokens=0,
                        output_tokens=0,
                        load_seconds=0.0,
                        ttft_p50_s=0.0,
                        ttft_p95_s=0.0,
                        total_p50_s=0.0,
                        total_p95_s=0.0,
                        throughput_p50_tok_s=0.0,
                        throughput_p95_tok_s=0.0,
                        peak_memory_mb=0.0,
                        skipped=True,
                        skip_reason=loaded.skip_reason,
                    )
                )
            continue

        for label, prompt in prompts.items():
            log.info("  prompt '%s'", label)
            input_tokens = count_input_tokens(loaded.tokenizer, prompt)
            metrics, peak_bytes, method = measure_config(
                loaded.model,
                loaded.tokenizer,
                prompt,
                device,
                max_new_tokens,
                warmup,
                runs,
            )
            result.cells.append(
                _aggregate(
                    config,
                    label,
                    input_tokens,
                    loaded.load_seconds,
                    metrics,
                    peak_bytes,
                    method,
                )
            )

        _free(loaded, device)

    return result
