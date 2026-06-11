"""Timing and memory measurement for a single config and prompt.

Honest accounting matters here:

* Time-to-first-token is the wall-clock gap between calling generate and the
  first decoded token arriving, measured with a streamer.
* Total latency is the wall-clock time for the whole generation.
* Throughput is output tokens divided by total latency.
* Generation is greedy and forced to exactly ``max_new_tokens`` (min equals
  max) so every run is directly comparable.
* Peak memory uses the CUDA allocator on CUDA, and resident set size sampled
  with psutil on MPS and CPU. Those two numbers do not mean the same thing,
  and the report says so.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil
import torch
from transformers import TextIteratorStreamer


@dataclass
class RunMetrics:
    ttft_s: float
    total_s: float
    output_tokens: int

    @property
    def throughput_tok_s(self) -> float:
        return self.output_tokens / self.total_s if self.total_s > 0 else 0.0


class _PeakSampler:
    """Poll a byte-count function in a thread and keep the maximum.

    Used on MPS and CPU, where there is no built-in peak counter. The sampled
    function is backend-specific (see ``measure_config``).
    """

    def __init__(self, source, interval: float = 0.02):
        self._source = source
        self._interval = interval
        self._peak = source()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._source())
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._peak = max(self._peak, self._source())

    @property
    def peak_bytes(self) -> int:
        return self._peak


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_inputs(tokenizer, prompt: str, device: str):
    """Apply the chat template and move tensors to the device."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs


def count_input_tokens(tokenizer, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids.shape[1]


@torch.no_grad()
def measure_once(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> RunMetrics:
    """Run one generation and return its metrics."""
    inputs = build_inputs(tokenizer, prompt, device)
    input_len = inputs.input_ids.shape[1]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
    )

    holder: dict = {}

    def _generate():
        holder["output"] = model.generate(**gen_kwargs)

    _sync(device)
    start = time.perf_counter()
    thread = threading.Thread(target=_generate)
    thread.start()

    ttft = None
    for _ in streamer:
        if ttft is None:
            ttft = time.perf_counter() - start
    thread.join()
    _sync(device)
    total = time.perf_counter() - start

    output_tokens = holder["output"].shape[1] - input_len
    if ttft is None:
        ttft = total
    return RunMetrics(ttft_s=ttft, total_s=total, output_tokens=output_tokens)


# How peak memory is counted on each backend. CUDA and MPS report allocator
# bytes (tensors on the device); CPU reports process resident set size. The
# meanings differ and the report labels which was used. Notably, on Apple
# Silicon the model weights live in Metal memory and do not appear in the
# process RSS at all, so RSS would badly undercount; the MPS allocator stat is
# the correct figure.
MEMORY_METHOD = {
    "cuda": "torch.cuda.max_memory_allocated",
    "mps": "torch.mps.current_allocated_memory (peak sampled)",
    "cpu": "psutil RSS (peak sampled)",
}


def measure_config(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    warmup: int,
    runs: int,
) -> tuple[list[RunMetrics], int, str]:
    """Run warmup (discarded) then measured runs.

    Returns the per-run metrics, the peak memory in bytes, and a label naming
    how that memory was counted. Peak is reset or freshly sampled after warmup
    so warmup allocation does not inflate it, then tracked across the measured
    runs.
    """
    for _ in range(warmup):
        measure_once(model, tokenizer, prompt, device, max_new_tokens)

    method = MEMORY_METHOD[device]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        metrics = [
            measure_once(model, tokenizer, prompt, device, max_new_tokens)
            for _ in range(runs)
        ]
        peak_bytes = torch.cuda.max_memory_allocated()
        return metrics, peak_bytes, method

    if device == "mps":
        torch.mps.empty_cache()
        source = torch.mps.current_allocated_memory
    else:
        proc = psutil.Process()
        source = lambda: proc.memory_info().rss  # noqa: E731

    with _PeakSampler(source) as sampler:
        metrics = [
            measure_once(model, tokenizer, prompt, device, max_new_tokens)
            for _ in range(runs)
        ]
    return metrics, sampler.peak_bytes, method
