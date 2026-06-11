# llm-laptop-bench

A small benchmark that measures what it actually costs to run an open
instruction-tuned language model on the hardware you already own. It reports
time-to-first-token, total generation latency, throughput, and peak memory
across precision and quantisation settings, on whichever backend is available:
CUDA, Apple Silicon (MPS), or plain CPU.

The aim is an honest, reproducible picture for practitioners deciding what runs
well on a laptop, not a leaderboard. The numbers below come from a real run on
the machine described in the results section.

## What it measures

For each combination of precision config and prompt length it records:

- Time-to-first-token, reported as p50 and p95 across the measured runs.
- Total generation latency, p50 and p95.
- Throughput in output tokens per second.
- Peak memory during the measured runs.
- Model load time, recorded separately and excluded from the timings above.

Generation is greedy and forced to exactly `max_new_tokens` (minimum equals
maximum), so every cell generates the same number of output tokens and the
runs are directly comparable. Warmup runs are excluded. A seed is set and the
hardware and library versions are written into every result file.

### A note on honesty

Peak memory is not measured the same way on every backend, because the
backends do not expose the same thing:

- On CUDA it is `torch.cuda.max_memory_allocated`, the high-water mark of the
  CUDA allocator. That is device memory used by tensors.
- On MPS and CPU it is the peak resident set size of the process, sampled with
  `psutil`. That includes the Python interpreter, the libraries, and the
  framework, not just the weights. On Apple Silicon, where CPU and GPU share
  one memory pool, resident set size is the closest honest figure available,
  but it is a larger and noisier number than the CUDA allocator stat. The two
  columns should not be compared across backends as if they were the same
  measurement.

Quantisation support also varies by backend. The benchmark attempts each
requested config and skips cleanly, logging the reason, when a backend cannot
run it (for example int4 weight packing where the kernels are absent). It does
not depend on bitsandbytes, which is CUDA-only.

## Models

The default model is
[`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct),
a small, ungated instruction-tuned model that fits comfortably on a laptop.

A documented alternative, also ungated, is
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
Select it with `--model Qwen/Qwen2.5-1.5B-Instruct`.

## Install

The project pins Python 3.13 and a fixed set of dependencies. Using
[uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.13 .venv
uv pip install -e .
```

Or with a standard virtual environment and pip:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```sh
python -m bench run
```

That runs the default sweep: configs `bf16,int8,int4`, two prompt lengths,
`max_new_tokens=128`, two warmup runs, ten measured runs. The backend is
detected automatically.

Common overrides:

```sh
# Force a backend
python -m bench run --device cpu

# A quicker, noisier pass
python -m bench run --configs bf16 --runs 5 --warmup 1 --max-new-tokens 64

# Use the TOML defaults file
python -m bench run --config bench.toml

# A different model
python -m bench run --model Qwen/Qwen2.5-1.5B-Instruct
```

Results are written to `results/` as JSON (self-describing, with hardware and
versions), CSV (the same rows, flat), and a Markdown table. The Markdown table
is also printed to the terminal.

## Configuration

Defaults can be set in a small TOML file and overridden on the command line.
See [`bench.toml`](bench.toml). CLI flags take precedence over the file, which
takes precedence over the built-in defaults.

## Results

<!-- RESULTS:START -->
Model `HuggingFaceTB/SmolLM2-1.7B-Instruct` on an Apple M3 Pro, 36 GB RAM,
macOS, device `mps`. Ten measured runs and two warmup runs per cell, seed 0,
`max_new_tokens=128`. Generation is greedy and forced to the full length.
Library versions: torch 2.12.0, transformers 5.11.0, optimum-quanto 0.2.7,
Python 3.13.13.

| Config | Prompt | In tok | Out tok | Load s | TTFT p50 (s) | TTFT p95 (s) | Total p50 (s) | Total p95 (s) | Tok/s p50 | Peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | short | 63 | 128 | 1.76 | 0.063 | 0.070 | 4.53 | 4.59 | 28.2 | 3302 |
| bf16 | long | 271 | 128 | 1.76 | 0.236 | 0.253 | 5.06 | 5.50 | 25.3 | 3355 |
| int8 | short | 63 | 128 | 1.82 | 0.237 | 0.269 | 27.81 | 28.84 | 4.6 | 3594 |
| int8 | long | 271 | 128 | 1.82 | 0.452 | 0.466 | 30.25 | 31.53 | 4.2 | 4479 |
| int4 † | short | 63 | 64 | 2.80 | 0.893 | 1.483 | 58.62 | 61.23 | 1.1 | 3706 |
| int4 † | long | 271 | 64 | 2.80 | 1.007 | 1.019 | 54.80 | 55.97 | 1.2 | 4048 |

Peak memory is the MPS allocator high-water mark
(`torch.mps.current_allocated_memory`), sampled across the measured runs.

† The int4 rows come from a separate run at a reduced profile (three measured
runs, one warmup, 64 output tokens) because at roughly one token per second the
full profile would take about three quarters of an hour and tell us nothing the
shorter run does not. Compare int4 on the per-token rate, not the total
latency. Raw files: [`results/sample-m3pro.json`](results/sample-m3pro.json)
(bf16 and int8) and [`results/probe-int4.json`](results/probe-int4.json).
<!-- RESULTS:END -->

## Trade-offs, and what surprised me

<!-- NARRATIVE:START -->
The result that matters most on this machine is the one I did not expect.
Quantisation made things worse, not better. On the M3 Pro, plain bf16 runs at
about 28 tokens per second. The same model quantised to int8 with quanto runs
at about 4.6, roughly six times slower, and int4 falls to about one token per
second. Neither quantised setting used less memory than bf16. int8 actually
used more.

That runs against the usual intuition, which is that lower precision means a
smaller and faster model. The intuition is shaped by CUDA, where quantised
matrix multiplications have dedicated kernels. On Apple Silicon there is no
such kernel for quanto's weight-only scheme, so the weights are stored at low
precision but every matrix multiply dequantises them back to bf16 before
computing. You pay the dequantisation cost on each step and keep bf16-sized
working memory, so you lose on speed and gain nothing on memory. int4 is worse
still because it relies on a C++ extension that runs partly on the CPU, which
is why it crawls. The honest conclusion for this hardware is that quanto
quantisation is for fitting a model that would otherwise not fit, not for
speeding up one that already does. If a model fits in bf16 on your laptop, run
it in bf16.

The second surprise was in the measurement itself. The brief asked for resident
memory via psutil on MPS, which is the sensible default on most systems. On
Apple Silicon it is misleading. When the model is moved to the MPS device its
weights are allocated in Metal memory, and that memory does not appear in the
process resident set size at all. Measured directly, the resident set size sat
at about 300 MB while the model occupied 3.3 GB of Metal memory. psutil would
have under-reported memory by an order of magnitude. The benchmark therefore
uses the MPS allocator high-water mark on MPS, the CUDA allocator on CUDA, and
psutil resident set size only on CPU, where the weights really do live in
ordinary system memory. The three numbers do not mean the same thing, and the
output labels which was used so they are not silently compared.

Beyond those two points the picture is unremarkable in a reassuring way. In
bf16 the model is responsive: time-to-first-token is around 60 milliseconds for
a short prompt and a little over 200 milliseconds for the longer one, the
difference being the cost of reading the larger prompt before the first token
appears. Sustained throughput drops only slightly with the longer prompt, from
28 to 25 tokens per second, which is what you would expect from a decode step
that is bound by memory bandwidth rather than by prompt length. Load time from
the local cache is under two seconds and is reported separately, so it never
contaminates the inference numbers. The variance across runs is small, with p95
close to p50, so the percentiles are doing little work here. That itself is
worth knowing: on a quiet laptop these figures are stable enough that a single
careful run tells you most of the story.

If you are choosing hardware or a precision setting for a model in this size
class, the practical reading is short. A recent Apple Silicon laptop runs a
1.7 billion parameter instruct model in bf16 comfortably and quickly, with
memory headroom to spare on a 36 GB machine. Quantisation through quanto is not
the lever to reach for here. On a CUDA machine, or for a model too large to fit
in bf16, the trade-off would look different, which is the whole point of
measuring on the hardware in front of you rather than trusting a single
headline number.
<!-- NARRATIVE:END -->

## Licence

MIT. See [LICENSE](LICENSE).
