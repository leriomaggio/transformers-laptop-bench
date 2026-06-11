"""Result writers: raw CSV and JSON, plus a Markdown table.

The JSON carries the full record including hardware and library versions so a
result is self-describing. The CSV is the same rows in a flat form. The
Markdown table is what goes into the README.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .hardware import Hardware
from .runner import SweepResult

_CSV_FIELDS = [
    "config",
    "prompt_label",
    "input_tokens",
    "output_tokens",
    "load_seconds",
    "ttft_p50_s",
    "ttft_p95_s",
    "total_p50_s",
    "total_p95_s",
    "throughput_p50_tok_s",
    "throughput_p95_tok_s",
    "peak_memory_mb",
    "memory_method",
    "skipped",
    "skip_reason",
]


def write_json(path: Path, sweep: SweepResult, hw: Hardware) -> None:
    payload = {
        "model_id": sweep.model_id,
        "device": sweep.device,
        "max_new_tokens": sweep.max_new_tokens,
        "warmup": sweep.warmup,
        "runs": sweep.runs,
        "seed": sweep.seed,
        "hardware": hw.as_dict(),
        "results": [c.as_dict() for c in sweep.cells],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, sweep: SweepResult) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for cell in sweep.cells:
            writer.writerow({k: cell.as_dict().get(k, "") for k in _CSV_FIELDS})


def render_markdown(sweep: SweepResult, hw: Hardware) -> str:
    lines: list[str] = []
    lines.append(f"## Results: {sweep.model_id}")
    lines.append("")
    lines.append(
        f"Device `{sweep.device}` on {hw.chip}, {hw.total_ram_gb} GB RAM. "
        f"{sweep.runs} measured runs, {sweep.warmup} warmup, "
        f"max_new_tokens={sweep.max_new_tokens}, seed {sweep.seed}."
    )
    lines.append("")
    lines.append(
        f"torch {hw.torch_version}, transformers {hw.transformers_version}, "
        f"optimum-quanto {hw.quanto_version}, Python {hw.python_version}."
    )
    lines.append("")
    header = (
        "| Config | Prompt | In tok | Out tok | Load s | "
        "TTFT p50 | TTFT p95 | Total p50 | Total p95 | "
        "Tok/s p50 | Peak MB |"
    )
    sep = "| " + " | ".join(["---"] * 11) + " |"
    lines.append(header)
    lines.append(sep)
    for c in sweep.cells:
        if c.skipped:
            lines.append(
                f"| {c.config} | {c.prompt_label} | - | - | - | - | - | "
                f"- | - | - | skipped |"
            )
            continue
        lines.append(
            f"| {c.config} | {c.prompt_label} | {c.input_tokens} | "
            f"{c.output_tokens} | {c.load_seconds:.2f} | "
            f"{c.ttft_p50_s:.3f} | {c.ttft_p95_s:.3f} | "
            f"{c.total_p50_s:.3f} | {c.total_p95_s:.3f} | "
            f"{c.throughput_p50_tok_s:.1f} | {c.peak_memory_mb:.0f} |"
        )

    methods = sorted({c.memory_method for c in sweep.cells if c.memory_method})
    if methods:
        lines.append("")
        lines.append(
            "Peak memory counted via " + "; ".join(methods) + "."
        )

    skipped = [c for c in sweep.cells if c.skipped]
    if skipped:
        lines.append("")
        lines.append("Skipped configs:")
        lines.append("")
        seen: set[str] = set()
        for c in skipped:
            if c.config in seen:
                continue
            seen.add(c.config)
            lines.append(f"- `{c.config}`: {c.skip_reason}")
    lines.append("")
    return "\n".join(lines)


def write_all(out_dir: Path, stem: str, sweep: SweepResult, hw: Hardware) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    write_json(json_path, sweep, hw)
    write_csv(csv_path, sweep)
    md_path.write_text(render_markdown(sweep, hw) + "\n")
    return {"json": json_path, "csv": csv_path, "md": md_path}
