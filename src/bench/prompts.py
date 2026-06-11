"""Fixed prompts used across every run.

Two lengths let us see how time-to-first-token scales with context. The exact
token count after chat templating is measured at run time and recorded, so the
labels here are nominal.
"""

from __future__ import annotations

SHORT_PROMPT = (
    "Explain in plain language what a large language model is and why running "
    "one locally on a laptop is different from calling a cloud API. Keep it "
    "to a short paragraph."
)

# A longer passage to push the input context up. The model is asked to
# summarise it, which keeps the task realistic rather than synthetic padding.
_PASSAGE = (
    "Consumer laptops have become capable enough to run small language models "
    "without a discrete accelerator, but the experience varies widely across "
    "hardware. A unified-memory Apple Silicon machine shares its RAM between "
    "the CPU and the GPU, so the same pool that holds the operating system and "
    "the browser also holds the model weights and the key-value cache. A "
    "desktop with a dedicated graphics card keeps model memory separate from "
    "system memory, which changes how headroom is measured and reported. On a "
    "CPU-only machine everything competes for the same cores, and throughput "
    "falls accordingly. Quantisation reduces the precision of the stored "
    "weights so that a model occupies less memory and, on some backends, runs "
    "faster, at the cost of a small and usually acceptable drop in output "
    "quality. The trade-offs are not uniform: a quantisation scheme that "
    "speeds up inference on one backend may simply use less memory on another, "
    "or may not be supported at all. Honest measurement therefore has to name "
    "the backend, the precision, and the way memory was counted, because a "
    "single tokens-per-second figure means little without that context."
)

LONG_PROMPT = (
    "Read the following passage and summarise it in three sentences for a "
    "practitioner deciding what hardware to buy.\n\n" + _PASSAGE
)


def get_prompts() -> dict[str, str]:
    """Map of nominal length label to prompt text."""
    return {"short": SHORT_PROMPT, "long": LONG_PROMPT}
