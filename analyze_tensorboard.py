#!/usr/bin/env python3
"""Summarize the steady-state FastSAC learner latency from TensorBoard."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def find_event_file(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        raise SystemExit(f"no TensorBoard event file found under {log_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--tag", default="timing/learner_train_ms")
    parser.add_argument("--tail", type=int, default=150)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    event_file = find_event_file(args.log_dir)
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()
    events = accumulator.Scalars(args.tag)
    if not events:
        raise SystemExit(f"tag {args.tag!r} has no samples in {event_file}")
    tail_events = events[-args.tail :]
    values = [float(event.value) for event in tail_events]
    summary = {
        "log_dir": str(args.log_dir.resolve()),
        "event_file": str(event_file.resolve()),
        "tag": args.tag,
        "total_samples": len(events),
        "tail_samples": len(values),
        "first_step": int(tail_events[0].step),
        "last_step": int(tail_events[-1].step),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "target_ms": args.target_ms,
        "passes_target": statistics.fmean(values) <= args.target_ms,
    }
    payload = json.dumps(summary, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
