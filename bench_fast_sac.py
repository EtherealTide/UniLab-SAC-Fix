#!/usr/bin/env python3
"""Reproduce the FastSAC learner update window from UniLab issue #662."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from uni_rl.algos.fast_sac.learner import FastSACLearner


@dataclass(frozen=True)
class Case:
    name: str
    use_compile: bool
    critic_graph: bool
    actor_graph: bool
    packed: bool
    force_compile_cudagraphs: bool | None = None


CASES = {
    case.name: case
    for case in (
        Case("eager", False, False, False, False),
        Case("compile_legacy", True, False, False, False, False),
        Case("compile_cudagraphs", True, False, False, False, True),
        Case("compile", True, False, False, False),
        Case("graph", True, True, True, False),
        Case("graph_packed", True, True, True, True),
    )
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_large_batch(
    *,
    batch_size: int,
    updates_per_step: int,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    device: torch.device,
    packed: bool,
    input_layout: str,
) -> dict[str, torch.Tensor]:
    rows = batch_size * updates_per_step
    if input_layout == "replay_packed_views" and not packed:
        width = 2 * obs_dim + action_dim + 3 + 2 * critic_obs_dim
        storage = torch.randn(rows, width, device=device)
        c = 0
        obs_sl = slice(c, c + obs_dim)
        c += obs_dim
        next_obs_sl = slice(c, c + obs_dim)
        c += obs_dim
        actions_sl = slice(c, c + action_dim)
        c += action_dim
        reward_col, done_col, truncated_col = c, c + 1, c + 2
        c += 3
        critic_sl = slice(c, c + critic_obs_dim)
        c += critic_obs_dim
        next_critic_sl = slice(c, c + critic_obs_dim)
        storage[:, done_col].zero_()
        storage[:, truncated_col].zero_()
        return {
            "obs": storage[:, obs_sl],
            "next_obs": storage[:, next_obs_sl],
            "actions": storage[:, actions_sl],
            "rewards": storage[:, reward_col],
            "dones": storage[:, done_col],
            "truncated": storage[:, truncated_col],
            "critic": storage[:, critic_sl],
            "next_critic": storage[:, next_critic_sl],
        }
    batch = {
        "obs": torch.randn(rows, obs_dim, device=device),
        "critic": torch.randn(rows, critic_obs_dim, device=device),
        "actions": torch.empty(rows, action_dim, device=device).uniform_(-1.0, 1.0),
        "rewards": torch.randn(rows, device=device),
        "next_obs": torch.randn(rows, obs_dim, device=device),
        "next_critic": torch.randn(rows, critic_obs_dim, device=device),
        "dones": torch.zeros(rows, device=device),
        "truncated": torch.zeros(rows, device=device),
    }
    if packed:
        keys = (
            "obs",
            "critic",
            "actions",
            "rewards",
            "next_obs",
            "next_critic",
            "dones",
            "truncated",
        )
        batch["sac_graph_packed_source"] = torch.cat(
            [batch[key].reshape(rows, -1) for key in keys], dim=1
        )
    return batch


def slice_batch(
    large_batch: dict[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in large_batch.items()}


def time_cuda_call(fn: Callable[[], None]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    host_start = time.perf_counter()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - host_start) * 1000.0


def run_case(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    learner = FastSACLearner(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        critic_obs_dim=args.critic_obs_dim,
        device=str(device),
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        num_atoms=args.num_atoms,
        num_q_networks=args.num_q_networks,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        use_compile=case.use_compile and case.force_compile_cudagraphs is None,
        use_cuda_graph_critic=case.critic_graph,
        use_cuda_graph_actor=case.actor_graph,
        use_cuda_graph_critic_packed_staging=case.packed,
        use_cuda_graph_actor_packed_staging=case.packed,
    )
    if case.force_compile_cudagraphs is not None:
        compile_kwargs = {"options": {"triton.cudagraphs": case.force_compile_cudagraphs}}
        learner._critic_loss_tensors = torch.compile(learner._critic_loss_tensors, **compile_kwargs)
        learner._actor_loss_tensors = torch.compile(learner._actor_loss_tensors, **compile_kwargs)
        learner.use_compile = True
    large_batch = make_large_batch(
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        obs_dim=args.obs_dim,
        critic_obs_dim=args.critic_obs_dim,
        action_dim=args.action_dim,
        device=device,
        packed=case.packed,
        input_layout=args.input_layout,
    )

    target_captured = bool(getattr(learner, "cuda_graph_critic_captures_target_update", False))

    critic_method = learner.update_critic_cuda_graph if case.critic_graph else learner.update_critic
    actor_method = learner.update_actor_cuda_graph if case.actor_graph else learner.update_actor
    critic_accepts_read_metrics = "read_metrics" in inspect.signature(critic_method).parameters
    actor_accepts_read_metrics = "read_metrics" in inspect.signature(actor_method).parameters

    def invoke_update(
        method: Callable[..., Any],
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool,
        accepts_read_metrics: bool,
    ) -> None:
        # The initial 77450d2 learner predates the read_metrics API.  Keep the
        # benchmark usable for both the historical baseline and the optimized
        # checkout instead of changing the old learner solely for benchmarking.
        if accepts_read_metrics:
            method(batch, read_metrics=read_metrics)
        else:
            method(batch)

    def critic_update(batch: dict[str, torch.Tensor], *, read_metrics: bool = False) -> None:
        invoke_update(
            critic_method,
            batch,
            read_metrics=read_metrics,
            accepts_read_metrics=critic_accepts_read_metrics,
        )

    def actor_update(batch: dict[str, torch.Tensor], *, read_metrics: bool = False) -> None:
        invoke_update(
            actor_method,
            batch,
            read_metrics=read_metrics,
            accepts_read_metrics=actor_accepts_read_metrics,
        )

    def update_window() -> None:
        for update_idx in range(args.updates_per_step):
            start = update_idx * args.batch_size
            batch = slice_batch(large_batch, start, start + args.batch_size)
            critic_update(
                batch,
                read_metrics=(
                    args.read_production_metrics and update_idx == args.updates_per_step - 1
                ),
            )
            if update_idx % args.policy_frequency == 0:
                actor_update(
                    batch,
                    read_metrics=False,
                )
            if not target_captured:
                learner.soft_update_target()
        if args.read_production_metrics:
            read_deferred_metrics = getattr(learner, "read_deferred_actor_metrics", None)
            if callable(read_deferred_metrics):
                read_deferred_metrics()

    # First call materializes optimizer state, compiles functions, and captures graphs.
    update_window()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        update_window()
        if args.inter_cycle_sleep_ms > 0:
            time.sleep(args.inter_cycle_sleep_ms / 1000.0)
    torch.cuda.synchronize()

    cycle_device_ms: list[float] = []
    cycle_host_ms: list[float] = []
    for _ in range(args.iterations):
        device_ms, host_ms = time_cuda_call(update_window)
        cycle_device_ms.append(device_ms)
        cycle_host_ms.append(host_ms)
        if args.inter_cycle_sleep_ms > 0:
            time.sleep(args.inter_cycle_sleep_ms / 1000.0)

    # Isolate the steady-state graph/eager call boundary after the cycle measurement.
    probe_batch = slice_batch(large_batch, 0, args.batch_size)
    critic_device_ms = [time_cuda_call(lambda: critic_update(probe_batch))[0] for _ in range(20)]
    actor_device_ms = [time_cuda_call(lambda: actor_update(probe_batch))[0] for _ in range(20)]

    result = {
        "case": asdict(case),
        "effective": {
            "use_compile": learner.use_compile,
            "use_amp": learner.use_amp,
            "amp_dtype": str(learner._amp_dtype),
            "critic_graph": learner.use_cuda_graph_critic,
            "actor_graph": learner.use_cuda_graph_actor,
            "critic_packed": learner.use_cuda_graph_critic_packed_staging,
            "actor_packed": learner.use_cuda_graph_actor_packed_staging,
            "target_update_captured": target_captured,
        },
        "cycle_device_ms": summary_ms(cycle_device_ms),
        "cycle_host_ms": summary_ms(cycle_host_ms),
        "critic_device_ms": summary_ms(critic_device_ms),
        "actor_device_ms": summary_ms(actor_device_ms),
        "target_ms": args.target_ms,
        "passes_target": summary_ms(cycle_host_ms)["median"] <= args.target_ms,
    }
    print(json.dumps(result, indent=2), flush=True)

    del learner, large_batch, probe_batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eager,compile,graph,graph_packed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--updates-per-step", type=int, default=8)
    parser.add_argument("--policy-frequency", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=98)
    parser.add_argument("--critic-obs-dim", type=int, default=101)
    parser.add_argument("--action-dim", type=int, default=29)
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--critic-hidden-dim", type=int, default=768)
    parser.add_argument("--num-atoms", type=int, default=101)
    parser.add_argument("--num-q-networks", type=int, default=2)
    parser.add_argument(
        "--input-layout",
        choices=("separate_contiguous", "replay_packed_views"),
        default="separate_contiguous",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--inter-cycle-sleep-ms", type=float, default=0.0)
    parser.add_argument(
        "--read-production-metrics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unilab-rl-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(selected) - CASES.keys())
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision("high")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_properties": str(torch.cuda.get_device_properties(0)),
        "unilab_rl_git": git_rev(args.unilab_rl_root.resolve()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "TORCH_LOGS")
            if os.environ.get(key) is not None
        },
    }
    print(json.dumps(metadata, indent=2), flush=True)
    results = [run_case(CASES[name], args) for name in selected]
    payload = {"metadata": metadata, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
