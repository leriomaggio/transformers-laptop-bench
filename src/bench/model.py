"""Model loading and precision/quantisation handling.

Load time is measured here and kept separate from inference timings. A config
names a precision: a floating-point baseline (bf16/fp16/fp32) or a quanto
weight-only quantisation (int8/int4). Quantisation paths that the active
backend cannot support are skipped cleanly, with the reason returned to the
caller for logging.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("bench.model")


def _ensure_ninja_on_path() -> None:
    """Make the interpreter's bin directory visible on PATH.

    int4 quantisation builds a small C++ extension through torch, which shells
    out to ``ninja``. When the package is installed in a virtual environment
    that has not been activated, that binary sits in the same directory as the
    Python executable but is not on PATH, and the build fails. Adding it makes
    int4 work whether or not the environment was activated.
    """
    bindir = os.path.dirname(sys.executable)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir and bindir not in parts:
        os.environ["PATH"] = os.pathsep.join([bindir, *parts])

# Configs the sweep understands. "kind" is either a float dtype or a quanto
# integer width.
FLOAT_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
QUANT_WIDTHS = {"int8", "int4"}


@dataclass
class LoadResult:
    model: object
    tokenizer: object
    load_seconds: float
    config: str
    skipped: bool = False
    skip_reason: str | None = None


class SkipConfig(Exception):
    """Raised when a config cannot run on the active backend."""


def _base_float_dtype(device: str) -> torch.dtype:
    """Compute dtype used as the base before quantisation."""
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def _apply_quanto(model, width: str, device: str):
    """Quantise a loaded model in place with optimum-quanto.

    Raises SkipConfig if quanto is unavailable or the width is not usable on
    this backend.
    """
    try:
        from optimum.quanto import freeze, qint4, qint8, quantize
    except ImportError as exc:
        raise SkipConfig(f"optimum-quanto not installed ({exc})") from exc

    weights = qint8 if width == "int8" else qint4
    if width == "int4":
        _ensure_ninja_on_path()

    # int4 weight packing relies on kernels that are not present on every
    # backend. Attempt it and surface a clean skip if the backend rejects it.
    try:
        quantize(model, weights=weights)
        freeze(model)
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        raise SkipConfig(
            f"{width} not supported on {device} backend ({exc})"
        ) from exc
    return model


def load(model_id: str, config: str, device: str) -> LoadResult:
    """Load tokenizer and model for a given precision config.

    Returns a LoadResult; on an unsupported config it carries skipped=True and
    a reason rather than raising, so the sweep can continue.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_quant = config in QUANT_WIDTHS
    base_dtype = _base_float_dtype(device) if is_quant else FLOAT_DTYPES[config]

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=base_dtype,
        low_cpu_mem_usage=True,
    )

    try:
        if is_quant:
            # Place on device first so quanto sees the target backend.
            model.to(device)
            _apply_quanto(model, config, device)
        else:
            model.to(device)
    except SkipConfig as exc:
        del model
        return LoadResult(
            model=None,
            tokenizer=tokenizer,
            load_seconds=0.0,
            config=config,
            skipped=True,
            skip_reason=str(exc),
        )

    model.eval()
    load_seconds = time.perf_counter() - start
    return LoadResult(
        model=model,
        tokenizer=tokenizer,
        load_seconds=load_seconds,
        config=config,
    )
