# UniLab FastSAC 单卡性能回退复现

本仓库保存 UniLab FastSAC 单卡训练性能回退的复现脚本、最新版补丁和实验结果。
正式验收对象是 RTX 4090 上的 G1WalkFlat/MuJoCo 真实训练，指标为 TensorBoard
`timing/learner_train_ms` 的最后 150 个样本均值，目标为 `<= 20 ms`。

最新对称 A/B（各 3 次、每次 300 iterations）结果：

```text
upstream f42570d       27.168 ± 0.377 ms
f42570d + 本仓库补丁    18.289 ± 0.217 ms
延迟下降                 32.68%
```

技术分析、逐次结果和 profiler 证据见 [REPORT.md](REPORT.md)。

## 三条路径的关系

| 路径 | 实际实现 | 在本报告中的用途 |
| --- | --- | --- |
| PR #667 手工 CUDA Graph | 显式打开 `use_cuda_graph_critic/actor`；捕获 eager、未充分融合的更新 | 历史方案和诊断对照；旧微基准约 29 ms，不是当前生产基线 |
| 最新 upstream baseline | `use_compile=true`，四个手工 Graph 开关均为 false，并强制 `triton.cudagraphs=False` | 本次正式 baseline，commit `f42570d` |
| 改进版 | `torch.compile`/Inductor 先融合，再用 Inductor CUDA Graph replay；手工 Graph 开关仍为 false | 本次正式 final，`f42570d` 加本仓库 patch |

因此，本次不是简单地重新打开 PR #667 的手工 Graph。修复目标是让当前生产使用的
`torch.compile` 路径形成稳定、可 replay 的热区，并清理其中的 CPU/GPU 同步边界。

背景：[UniLab issue #662](https://github.com/Motphys/UniLab/issues/662)、
[UniLab PR #667](https://github.com/Motphys/UniLab/pull/667)。

## 仓库内容

```text
README.md                         复现实验
REPORT.md                         技术报告
bench_fast_sac.py                 learner 微基准
analyze_tensorboard.py            正式训练统计脚本
patches/                          基于最新 upstream 的 Git patch
results/                          历史与最新 JSON 结果
```

## 1. 准备 baseline 与 final

以下示例假定三个项目位于同一个目录。先把变量改成自己的绝对路径：

```bash
UNILABSIM_ROOT=/path/to/unilabsim
SAC_FIX_DIR="$UNILABSIM_ROOT/UniLab-SAC-Fix"
```

从最新版基线 `f42570d` 建立两份独立 worktree，并只给 final 应用补丁：

```bash
cd "$UNILABSIM_ROOT/unilab_rl"
git fetch upstream
git worktree add --detach /tmp/unilab-rl-sac-baseline f42570d1c27326bf61a4b4ed598d7eeac8f1225d
git worktree add --detach /tmp/unilab-rl-sac-final f42570d1c27326bf61a4b4ed598d7eeac8f1225d
git -C /tmp/unilab-rl-sac-final am \
  "$SAC_FIX_DIR/patches/0001-perf-fast-sac-restore-fused-single-GPU-update-path.patch"
```

确认版本和实际 import 路径：

```bash
git -C /tmp/unilab-rl-sac-baseline rev-parse HEAD
git -C /tmp/unilab-rl-sac-final rev-parse HEAD

PYTHONPATH=/tmp/unilab-rl-sac-baseline/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python -c "import uni_rl; print(uni_rl.__file__)"

PYTHONPATH=/tmp/unilab-rl-sac-final/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python -c "import uni_rl; print(uni_rl.__file__)"
```

baseline 应为 `f42570d...`；本仓库 patch 的 preflight commit 为 `65dda3c...`。
两条 import 输出必须分别指向对应 worktree，而不是 `.venv/site-packages`。

## 2. 运行真实训练

测试环境与正式结果保持一致：RTX 4090、2048 environments、batch 8192、每周期
8 次 critic update、`policy_frequency=4`（2 次 actor update）、AMP、300 iterations。
测试时避免其他进程占用 GPU，并为每次运行使用全新的日志目录。

在 UniLab 根目录定义下列函数：

```bash
cd "$UNILABSIM_ROOT/UniLab"

run_sac_test() {
  local rl_root="$1"
  local log_dir="$2"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$rl_root/src:$UNILABSIM_ROOT/UniLab/src:$UNILABSIM_ROOT/unisim/src" \
  UV_CACHE_DIR=/tmp/unilab-uv-cache \
  uv run --offline train \
    --algo sac --task g1_walk_flat --sim mujoco \
    'training.devices=[0]' training.no_play=true training.export_onnx=false \
    training.trace_enabled=false training.nvtx_profile_ranges=false \
    training.log_dir="$log_dir" \
    algo.max_iterations=300 algo.save_interval=100000 algo.num_envs=2048 \
    algo.batch_size=8192 algo.replay_buffer_n=512 algo.updates_per_step=8 \
    algo.policy_frequency=4 algo.actor_hidden_dim=512 algo.critic_hidden_dim=768 \
    algo.num_atoms=101 training.use_amp=true algo.algo_params.amp_dtype=auto \
    algo.algo_params.use_compile=true \
    algo.algo_params.use_cuda_graph_critic=false \
    algo.algo_params.use_cuda_graph_actor=false \
    algo.algo_params.use_cuda_graph_critic_packed_staging=false \
    algo.algo_params.use_cuda_graph_actor_packed_staging=false
}
```

baseline 与 final 各运行三次：

```bash
for run_index in 1 2 3; do
  run_sac_test /tmp/unilab-rl-sac-baseline "/tmp/sac_latest_benchmark/baseline_run${run_index}"
done
for run_index in 1 2 3; do
  run_sac_test /tmp/unilab-rl-sac-final "/tmp/sac_latest_benchmark/final_run${run_index}"
done
```

训练结束后检查 final 的真实运行路径：

```bash
jq '{status, completed_iterations, cuda_graph: .runtime_manifest.cuda_graph}' \
  /tmp/sac_latest_benchmark/final_run1/run_summary.json
```

关键字段应为：手工 `critic_enabled/actor_enabled=false`、
`inductor_critic_cudagraphs/inductor_actor_cudagraphs=true`、
`device_finite_optimizer_gating=true`、`fallback_reasons=[]`。

## 3. 统计正式结果

```bash
for version in baseline final; do
  for run_index in 1 2 3; do
    UV_CACHE_DIR=/tmp/unilab-uv-cache \
    uv run --offline python "$SAC_FIX_DIR/analyze_tensorboard.py" \
      "/tmp/sac_latest_benchmark/${version}_run${run_index}" \
      --tail 150 --target-ms 20 \
      --output "/tmp/${version}_run${run_index}.json"
  done
done
```

每次应有 `total_samples=300`、`tail_samples=150`。验收口径是 `mean_ms <= 20`；
median、p90 和 p95 用于描述分布。

## 4. 可选 learner 微基准

微基准用于快速验证路径，不替代真实训练。对 baseline/final 分别替换 `rl_root`：

```bash
rl_root=/tmp/unilab-rl-sac-final
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$rl_root/src" \
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline python "$SAC_FIX_DIR/bench_fast_sac.py" \
  --cases compile --input-layout replay_packed_views --warmup 20 --iterations 100 \
  --inter-cycle-sleep-ms 32 --unilab-rl-root "$rl_root" \
  --output /tmp/sac-micro-final.json
```

若要观察 PR #667 风格的手工 Graph，将 `--cases compile` 改为 `--cases graph_packed`。
该路径不是正式 baseline，不能混入正式 A/B 结论。
