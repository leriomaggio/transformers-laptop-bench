"""Backend detection and environment capture.

The benchmark records exactly what it ran on so that numbers can be read in
context. Device selection prefers CUDA, then Apple Silicon (MPS), then CPU,
and degrades quietly when a backend is absent.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass, asdict

import psutil
import torch


@dataclass
class Hardware:
    device: str
    chip: str
    cpu_count: int
    total_ram_gb: float
    platform: str
    python_version: str
    torch_version: str
    transformers_version: str
    quanto_version: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def select_device(requested: str = "auto") -> str:
    """Return the device string to run on.

    "auto" picks the best available backend. An explicit request is honoured
    when the backend is present, and falls back to CPU with the reason logged
    by the caller otherwise.
    """
    if requested != "auto":
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _chip_name() -> str:
    """Best effort human-readable processor name."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            )
            return out.strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return platform.processor() or platform.machine()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(device: str) -> Hardware:
    return Hardware(
        device=device,
        chip=_chip_name(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        total_ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=_version("transformers") or "unknown",
        quanto_version=_version("optimum-quanto"),
    )
