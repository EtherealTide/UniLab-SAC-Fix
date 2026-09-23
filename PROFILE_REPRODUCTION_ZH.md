# SAC 真实训练 profiling 复现实验

本文只用于采集报告证据，不替代正式性能验收。正式性能必须使用
`training.trace_enabled=false` 的 300-iteration 训练；trace 会引入少量记录开销。

## 1. 当前已打通的采集链

UniLab/unilab_rl 已有真实 off-policy trace recorder。打开下面两个配置后，runner
会在真实 G1/MuJoCo SAC 进程中记录 learner、replay、inference、critic/actor update
以及 CUDA Event 时间线：

```text
training.trace_enabled=true
training.trace_cuda_events=true
training.nvtx_profile_ranges=true
```

结束后必须存在：

```text
<log_dir>/run_summary.json
<log_dir>/run_config.json
<log_dir>/perfetto_offpolicy_timeline.json
```

其中 `run_summary.json.runtime_manifest.cuda_graph` 用来确认实际走的是哪条 Graph
路径；`perfetto_offpolicy_timeline.json` 是真实训练的 Chrome/Perfetto trace，
不是 learner 微基准。

当前机器没有安装 `nsys`（`command -v nsys` 无输出），因此本机先使用上述
应用级 Perfetto/CUDA Event trace。它能证明真实训练中的阶段耗时和粗粒度 CPU/GPU
时间线，但不是逐 kernel 的 Nsight 数据；NVTX 标记只有在 Nsight 中才能用于
`cudaLaunchKernel`/`cudaGraphLaunch` 和逐 kernel 统计。
若 mentor 明确要求 Nsight Systems 的 kernel/API 统计，需要在安装了 Nsight
Systems 的机器上额外执行第 5 节。

## 2. Profiling smoke（每个版本一次）

下面命令只跑 30 iteration，用来验证流程。baseline 和 final 的 workload 必须完全
一致，唯一变化是 `unilab_rl` 的 `PYTHONPATH` 和输出目录。

### baseline `77450d2`

```bash
cd /home/pc823/桌面/unilabsim/UniLab
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/unilab-rl-sac-baseline/src:/home/pc823/桌面/unilabsim/UniLab/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 algo.max_iterations=30 algo.save_interval=100000 \
  algo.batch_size=8192 algo.replay_buffer_n=512 algo.updates_per_step=8 \
  algo.policy_frequency=4 algo.actor_hidden_dim=512 algo.critic_hidden_dim=768 \
  algo.num_atoms=101 training.devices='[0]' training.no_play=true \
  training.export_onnx=false training.log_dir=/tmp/sac-profile-baseline \
  training.trace_enabled=true training.trace_output_dir=/tmp/sac-profile-baseline \
  training.trace_cuda_events=true training.nvtx_profile_ranges=true \
  algo.algo_params.amp_dtype=auto algo.algo_params.use_compile=true \
  algo.algo_params.use_cuda_graph_critic=false \
  algo.algo_params.use_cuda_graph_actor=false \
  algo.algo_params.use_cuda_graph_critic_packed_staging=false \
  algo.algo_params.use_cuda_graph_actor_packed_staging=false
```

### final `c543650`

将上面命令中的两处替换为：

```text
PYTHONPATH=/home/pc823/桌面/unilabsim/unilab_rl/src:/home/pc823/桌面/unilabsim/UniLab/src
training.log_dir=/tmp/sac-profile-final
training.trace_output_dir=/tmp/sac-profile-final
```

## 3. 分析 trace

```bash
cd /home/pc823/桌面/unilabsim/UniLab
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline \
  scripts/analyze_offpolicy_trace.py \
  /tmp/sac-profile-final/perfetto_offpolicy_timeline.json \
  --steps-per-cycle 2048 --drop-first 5 --training-e2e \
  --event learner/update_phase \
  --event learner/update_critic \
  --event learner/update_actor \
  --event learner/replay_sample \
  --event learner/inference \
  --gap-event learner/update_critic \
  --gap-event learner/update_actor
```

`--drop-first 5` 是按 cycle 统计丢弃编译/启动阶段。报告中应将 profiling 数据标为
“稳态 trace（剔除前 5 个 cycle）”，不能把它当作未 profiling 的正式训练均值。

流程验证时，baseline/final 的 `learner/update_phase` median 分别约为 26.8 ms 和
17.1 ms；这只是 30-iteration profiling smoke，最终报告必须换成用户自己重跑的
正式数据。

## 4. 需要提交给报告撰写者的数据

每个 baseline/final profiling run 请发送：

1. `run_summary.json`；
2. `run_config.json`；
3. `perfetto_offpolicy_timeline.json`，或完整的 analyzer 输出；
4. 下面命令的输出：

   ```bash
   git -C /tmp/unilab-rl-sac-baseline rev-parse HEAD  # baseline
   git -C /home/pc823/桌面/unilabsim/unilab_rl rev-parse HEAD  # final
   ```

5. `python -c` 不要直接使用；统一使用 `uv run --offline` 检查实际 import：

   ```bash
   PYTHONPATH=<对应 unilab_rl>/src:/home/pc823/桌面/unilabsim/UniLab/src \
   UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline python -c \
   "import uni_rl; print(uni_rl.__file__)"
   ```

## 5. 如果需要 Nsight Systems

先在目标机器确认：

```bash
nsys --version
```

然后在同一条真实训练命令外包一层（建议 30--100 iteration，避免 trace 过大）：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o /tmp/nsys-sac-final \
  uv run --offline train ...同一组 Hydra overrides...
```

导出重点统计：

```bash
nsys stats --report cudaapisum,gpukernsum,nvtxppsum --format csv \
  /tmp/nsys-sac-final.nsys-rep > /tmp/nsys-sac-final-stats.csv
```

报告至少保留：

- `cudaLaunchKernel` 与 `cudaGraphLaunch` 的调用次数和总时长；
- `critic/loss_compiled`、`critic/backward`、`actor/loss_compiled`、
  `actor/backward` 的 NVTX 区间；
- baseline/final 使用完全相同的 iteration、seed、GPU 和 Hydra overrides。

## 6. 正式性能数据不要和 profiling 混用

正式验收另开全新目录，关闭 trace，baseline/final 各跑 3 次 300 iteration：

```text
training.trace_enabled=false
training.nvtx_profile_ranges=false
```

每次只保留一个 TensorBoard event 文件，并发送 `analyze_tensorboard.py --tail 150`
的输出。最终报告会把：

```text
正式 learner mean/median/p90
+ 真实 trace 的 update_phase 与 NVTX/CUDA Event 证据
+ runtime_manifest 的实际 Graph 路径
```

放在同一张 baseline/final 对照表中。
