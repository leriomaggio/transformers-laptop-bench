"""Command-line entry point.

Usage:
    python -m bench run --model ... --configs bf16,int8,int4

Defaults match the "standard" profile. An optional TOML file supplies defaults
that CLI flags then override.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from . import hardware, report
from .prompts import get_prompts
from .runner import run_sweep

DEFAULTS = {
    "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "configs": "bf16,int8,int4",
    "device": "auto",
    "max_new_tokens": 128,
    "runs": 10,
    "warmup": 2,
    "seed": 0,
    "out_dir": "results",
    "stem": "run",
}


def _load_toml_defaults(path: Path) -> dict:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return data.get("bench", data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark sweep")
    run.add_argument("--model", help="Hugging Face model id")
    run.add_argument("--configs", help="comma list: bf16,fp16,fp32,int8,int4")
    run.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    run.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    run.add_argument("--runs", type=int, help="measured runs per cell")
    run.add_argument("--warmup", type=int, help="discarded warmup runs")
    run.add_argument("--seed", type=int)
    run.add_argument("--out-dir", dest="out_dir")
    run.add_argument("--stem", help="output filename stem")
    run.add_argument("--config", type=Path, help="optional TOML config file")
    return parser


def _resolve(args: argparse.Namespace) -> dict:
    settings = dict(DEFAULTS)
    if args.config:
        settings.update(_load_toml_defaults(args.config))
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    return settings


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _resolve(args)
    configs = [c.strip() for c in str(settings["configs"]).split(",") if c.strip()]

    device = hardware.select_device(settings["device"])
    if settings["device"] != "auto" and device != settings["device"]:
        logging.warning(
            "requested device '%s' unavailable, using '%s'",
            settings["device"],
            device,
        )
    hw = hardware.collect(device)
    logging.info("device=%s chip=%s ram=%sGB", device, hw.chip, hw.total_ram_gb)

    sweep = run_sweep(
        model_id=settings["model"],
        configs=configs,
        device=device,
        prompts=get_prompts(),
        max_new_tokens=int(settings["max_new_tokens"]),
        warmup=int(settings["warmup"]),
        runs=int(settings["runs"]),
        seed=int(settings["seed"]),
    )

    paths = report.write_all(
        Path(settings["out_dir"]), str(settings["stem"]), sweep, hw
    )
    print(report.render_markdown(sweep, hw))
    print(f"\nWrote {paths['json']}, {paths['csv']}, {paths['md']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
