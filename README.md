# UniLab SAC CUDA Graph regression investigation

This repository contains a reproducible benchmark for the FastSAC learner path
used by `sac/g1_walk_flat/mujoco` in UniLab issue #662 and PR #667.

The investigation and final numbers are documented in [RESULTS.md](RESULTS.md).
The step-by-step Chinese reproduction guide and final report are available in
[REPRODUCTION_REPORT_ZH.md](REPRODUCTION_REPORT_ZH.md).
The complete experiment-by-experiment optimization log is in
[EXPERIMENT_LOG_ZH.md](EXPERIMENT_LOG_ZH.md).
On an RTX 4090, three independent 300-iteration runs achieved tail-150 means of
19.177, 19.140, and 18.749 ms, meeting the `<=20 ms` target in every run.

The production `unilab_rl` change is included as a reviewable Git patch under
`patches/`. Apply it from an `unilab_rl` checkout with:

```bash
git am /path/to/UniLab-SAC-Fix/patches/0001-perf-fast-sac-restore-fused-single-GPU-update-path.patch
```

The primary acceptance metric is one learner update cycle with the effective
production shape: batch size 8192, eight critic updates, and two actor updates
(`policy_frequency=4`), with observation/critic/action widths 98/101/29. The
target on a single RTX 4090 is median host latency
at or below 20 ms after warmup. CUDA-event latency is also recorded so host
submission/synchronization regressions can be separated from device work.

Run against a local `unilab_rl` checkout:

```bash
cd /path/to/unilab_rl
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline \
  python /path/to/UniLab-SAC-Fix/bench_fast_sac.py \
  --unilab-rl-root "$PWD" \
  --output /path/to/UniLab-SAC-Fix/results/latest.json
```

The benchmark compares eager, `torch.compile`, CUDA Graph, and CUDA Graph with
packed staging. Graph cases retain the production update order and only skip
the external Polyak update when the learner reports that it is captured inside
the critic graph.

Summarize a full training log with the same acceptance window:

```bash
cd /path/to/UniLab
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline \
  /path/to/UniLab-SAC-Fix/analyze_tensorboard.py \
  /path/to/training-log --tail 150
```
