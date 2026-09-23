# UniLab SAC 单卡性能问题复现与分析报告

## 1. 先理解我们要测什么

这次验收对象不是一次普通的神经网络前向，而是一个完整的 SAC learner
更新周期：

- batch size：8192；
- 8 次 critic 更新；
- 每 4 次 critic 更新一次 actor，因此共有 2 次 actor 更新；
- observation / critic observation / action 维度分别为 98 / 101 / 29；
- BF16 AMP；
- 单张 RTX 4090。

完整训练使用 TensorBoard 指标 `timing/learner_train_ms`。本文采用的验收口径是：

> 运行 300 iterations，去掉前 150 次预热和编译影响，计算第 151–300 次的
> `learner_train_ms` 算术平均值，要求不超过 20 ms。

注意：这不等于要求每一次更新都低于 20 ms。最终实验的 p90 仍约为
22–24 ms。如果 mentor 要求的是 p90 或最坏值不超过 20 ms，需要重新定义验收
条件并继续优化。

## 2. 目录和版本

当前机器上的目录为：

```text
/home/pc823/桌面/unilabsim/
├── UniLab/                 # 环境、任务配置和训练入口
├── unilab_rl/              # SAC learner 和 off-policy runtime
├── unisim/                 # 仿真后端统一接口
└── UniLab-SAC-Fix/         # 本次基准、结果、报告和补丁
```

本次修复对应：

- `unilab_rl` 分支：`codex/sac-cuda-graph-fix`；
- `unilab_rl` commit：`c543650`；
- 复现实验仓库 commit：`5f872b4`（后续文档提交会在此基础上增加）；
- 修复补丁：
  `patches/0001-perf-fast-sac-restore-fused-single-GPU-update-path.patch`。

当前工作区已经包含修复，不要再次执行 `git am`。如果在另一份干净的
`unilab_rl` 上复现，可以执行：

```bash
cd /path/to/unilab_rl
git am /path/to/UniLab-SAC-Fix/patches/0001-perf-fast-sac-restore-fused-single-GPU-update-path.patch
```

检查版本：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl
git branch --show-current
git log -1 --oneline
```

期望看到 `codex/sac-cuda-graph-fix` 和 `c543650`。如果不是，先停止性能测试，
因为后面的数字将无法和本报告对应。

## 3. 第一步：确认 GPU 没有被别人占用

执行：

```bash
nvidia-smi
```

重点看三件事：

1. GPU 型号是 RTX 4090；
2. 没有其他训练进程占用大量显存或 GPU utilization；
3. 测试期间不要同时运行另一个 CUDA benchmark。

为什么先做这一步：完整训练只有约 19 ms，其他进程造成 1–2 ms 干扰就足以
让结果从通过变成不通过。我们曾在非独占状态观察到约 20.5 ms，而相同代码在
三次受控运行中均低于 20 ms。

如果 GPU 正忙，先等占用者结束，再继续；不要通过杀掉未知进程来“清场”。

## 4. 第二步：确认 Python 真正加载了本地修复代码

这是最容易忽略的一步。UniLab 的虚拟环境可能加载已经安装的
`unilab-rl 1.2.1`，而不是旁边的源码仓库。

执行：

```bash
cd /home/pc823/桌面/unilabsim/UniLab
PYTHONPATH=/home/pc823/桌面/unilabsim/unilab_rl/src:/home/pc823/桌面/unilabsim/unisim/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python -c "import torch, uni_rl; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(uni_rl.__file__)"
```

期望结果：

- PyTorch 类似 `2.8.0+cu128`；
- GPU 为 `NVIDIA GeForce RTX 4090`；
- `uni_rl.__file__` 位于
  `/home/pc823/桌面/unilabsim/unilab_rl/src/uni_rl/`。

如果最后一项指向 `.venv/site-packages`，说明还在测试旧包。此时不要继续；先
检查 `PYTHONPATH` 是否拼写正确。

## 5. 第三步：先阅读已有原始结果

这一步不运行 GPU，只验证我们将要复现的数量级：

```bash
cd /home/pc823/桌面/unilabsim/UniLab-SAC-Fix

jq -r '.results[] | [.case.name, .cycle_host_ms.mean, .cycle_host_ms.median, .cycle_host_ms.p90] | @tsv' \
  results/clean_main_compile_graph_packed.json

jq . results/training_repeats.json
```

第一份文件大致显示：

```text
compile       mean 19.43 ms    median 19.38 ms
graph_packed  mean 29.09 ms    median 29.14 ms
```

这里得到的第一个判断是：手工 CUDA Graph 并没有消失，但在当前 4090 上它捕获
的是未融合 eager kernels，反而比 `torch.compile` 更慢。因此后续主线不应该是
“强行打开 PR #667 的四个开关”，而应该检查 `torch.compile` 生成的图是否被拆碎。

这些保存结果的 warmup 等参数并不完全相同，只能作为方向性证据。正式 A/B
必须使用下一步中的同一条命令和同一组参数。

## 6. 第四步：运行 learner 微基准

执行优化后默认路径：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/pc823/桌面/unilabsim/unilab_rl/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/bench_fast_sac.py \
  --cases compile \
  --input-layout replay_packed_views \
  --warmup 20 \
  --iterations 100 \
  --inter-cycle-sleep-ms 32 \
  --unilab-rl-root /home/pc823/桌面/unilabsim/unilab_rl \
  --output /tmp/sac-micro-fixed.json
```

参数中的 32 ms sleep 用来近似真实训练中环境收集阶段的间隔。没有这个间隔，
GPU 会一直处于高频状态，微基准会比真实训练更乐观。

查看结果：

```bash
jq '.results[0] | {cycle_host_ms, critic_device_ms, actor_device_ms, passes_target}' \
  /tmp/sac-micro-fixed.json
```

本次参考结果：

```text
完整 learner cycle host mean    16.97 ms
完整 learner cycle host median  16.97 ms
critic 单次 mean                 1.70 ms
actor 单次 mean                  1.20 ms
passes_target                    true
```

如何决定下一步：

- 如果 median 在 17–19 ms，说明 learner 本体修复生效，可以进入完整训练；
- 如果仍在 24 ms 左右，优先重新检查 `uni_rl.__file__`；
- 如果第一次特别慢、后面正常，这是 `torch.compile` 首次编译，不能把第一次计入
  稳态结果；
- 如果 `graph_packed` 比 `compile` 慢，不是异常，这正是当前 4090 上的观察结果。

## 7. 可选：严格复现修复前后的 A/B

当前修复前基线 commit 是 `77450d2`。建议使用临时 worktree，避免切换或破坏当前
分支：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl
git worktree add /tmp/unilab-rl-sac-baseline 77450d2
```

然后把上一步命令中的 `PYTHONPATH` 和 `--unilab-rl-root` 都改为：

```text
/tmp/unilab-rl-sac-baseline/src
/tmp/unilab-rl-sac-baseline
```

其他参数必须完全不变，输出写到另一个 JSON。最后比较两个 JSON 的
`cycle_host_ms`。如果两边的 workload、warmup、sleep 或输入布局不同，就不能把
差值全部归因于代码修改。

## 8. 第五步：运行真正的 G1/MuJoCo 完整训练

每次使用新的日志目录，不要复用旧目录，否则一个目录中会存在多个 TensorBoard
event 文件，分析脚本可能读到错误的一次实验。

```bash
cd /home/pc823/桌面/unilabsim/UniLab

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/pc823/桌面/unilabsim/unilab_rl/src:/home/pc823/桌面/unilabsim/unisim/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline train \
  --algo sac --task g1_walk_flat --sim mujoco \
  'training.devices=[0]' \
  training.no_play=true \
  training.export_onnx=false \
  training.log_dir=/tmp/unilab-sac-reproduce-1 \
  algo.max_iterations=300 \
  algo.save_interval=100000 \
  algo.num_envs=2048 \
  algo.batch_size=8192 \
  algo.replay_buffer_n=512 \
  algo.updates_per_step=8 \
  algo.policy_frequency=4 \
  algo.actor_hidden_dim=512 \
  algo.critic_hidden_dim=768 \
  algo.num_atoms=101 \
  training.use_amp=true \
  algo.algo_params.amp_dtype=auto \
  algo.algo_params.use_compile=true \
  algo.algo_params.use_cuda_graph_critic=false \
  algo.algo_params.use_cuda_graph_actor=false \
  algo.algo_params.use_cuda_graph_critic_packed_staging=false \
  algo.algo_params.use_cuda_graph_actor_packed_staging=false
```

这里有一个看似矛盾但非常重要的点：四个手工 CUDA Graph 开关是 `false`，并不
表示完全没有 CUDA Graph。默认路径由 `torch.compile`/Inductor 负责融合 kernel，
然后由 Inductor 自己执行 CUDA Graph replay。

训练结束时应看到：

- `Iterations: 300/300`；
- `Training complete`；
- 终端的 `Train` 通常在 19 ms 左右。

终端最后一行只是一个瞬时或平滑值，不能直接作为最终报告数字。

## 9. 第六步：检查运行时确实走了正确路径

执行：

```bash
jq '{status, completed_iterations, cuda_graph: .runtime_manifest.cuda_graph}' \
  /tmp/unilab-sac-reproduce-1/run_summary.json
```

期望关键字段：

```json
{
  "status": "completed",
  "completed_iterations": 300,
  "cuda_graph": {
    "critic_enabled": false,
    "actor_enabled": false,
    "inductor_critic_cudagraphs": true,
    "inductor_actor_cudagraphs": true,
    "device_finite_optimizer_gating": true,
    "fallback_reasons": []
  }
}
```

判断逻辑：

- `critic_enabled/actor_enabled=false`：没有走较慢的手工 Graph；
- `inductor_*_cudagraphs=true`：融合后的计算图正在 replay；
- `fallback_reasons=[]`：没有因为 FP16 scaler 或 observation normalization 回退；
- `device_finite_optimizer_gating=true`：没有重新引入每轮 host finite-check 同步。

任一字段不符合时，先解释运行路径，不要急着比较毫秒数。

## 10. 第七步：计算正式验收指标

```bash
cd /home/pc823/桌面/unilabsim/UniLab

UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/analyze_tensorboard.py \
  /tmp/unilab-sac-reproduce-1 \
  --tail 150 \
  --target-ms 20
```

重点看：

- `total_samples` 应为 300；
- `tail_samples` 应为 150；
- `mean_ms <= 20` 才按本文口径通过；
- median 用于描述典型值；
- p90/p95 用于描述抖动，不要用较好看的最后 20 次替代 tail-150。

本次三次受控结果为：

| 运行 | tail-150 mean | median | p90 |
| --- | ---: | ---: | ---: |
| 1 | 19.177 ms | 18.941 ms | 23.857 ms |
| 2 | 19.140 ms | 19.416 ms | 22.428 ms |
| 3 | 18.749 ms | 18.961 ms | 22.556 ms |

三次 mean 的平均值为 19.022 ms。修复前完整训练约为 24.56 ms，因此平均降低约
22.6%。

建议把上述完整训练再执行三次，并分别使用
`/tmp/unilab-sac-reproduce-1/2/3`。只有三次 tail-150 mean 都通过，才报告为
“可重复达到 20 ms 以内”。

## 11. 第八步：运行正确性和质量门禁

性能优化不能以破坏数值正确性为代价。执行：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl

UV_CACHE_DIR=/tmp/unilab-uv-cache make format
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline mypy src/uni_rl
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline pyright
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/unilab-uv-cache \
  uv run --offline pytest --cov=src/uni_rl
```

本次结果：

- Ruff、Mypy、Pyright 全通过；
- 完整测试：405 passed，8 skipped；
- CUDA 聚焦回归：67 passed；
- finite、NaN loss、NaN gradient 和 NaN 后恢复 finite 的 CUDA Graph smoke 均通过；
- CUDA Graph replay 中随机噪声会继续推进，没有每次重放同一份随机数。

如果 Gloo 测试在受限沙箱中报“Cannot resolve 127.0.0.1”，这是沙箱网络限制；
应在允许本机回环网络的正常 shell 中重跑，而不是修改算法代码绕过测试。

## 12. 问题源代码地图

以下路径均相对于 `unilab_rl` 仓库。

### 12.1 actor 计算图与 critic 梯度：需要区分基线和中间实验

- `src/uni_rl/algos/fast_sac/learner.py:736`：`_actor_loss_tensors`，这是交给
  `torch.compile` 的 actor loss 主体；
- `src/uni_rl/algos/fast_sac/learner.py:656`：最终版新增的
  `_critic_parameters_frozen`；
- `src/uni_rl/algos/fast_sac/learner.py:767`、`1518`：最终版在调用 compiled
  callable 之前冻结 critic，而不是把冻结逻辑写进 compiled loss。

需要特别纠正一个容易造成误解的说法：基线 `77450d2` 的
`_actor_loss_tensors`（约 671--695 行）没有 `requires_grad_`；它的
`update_actor` 直接调用该函数。因此“基线把 `requires_grad_` 写进 compiled loss
导致 graph break”不是 Git 历史能够证明的事实，而是我们排查时曾尝试过的中间写法。
该中间写法确实会造成图拆分，所以随后把冻结边界移到 compiled callable 外；最终版
compiled loss 内只保留张量计算。冻结 critic 不是切断 actor 梯度：critic 权重不需要
`dL/dW`，但输入 action 仍然需要 `dQ/da`。

冻结 critic 不是切断 actor 梯度：critic 权重不需要 `dL/dW`，但输入 action 仍然
需要 `dQ/da`，所以策略仍能正常学习。

### 12.2 Inductor CUDA Graph 开关

- `src/uni_rl/algos/fast_sac/learner.py:399`：FastSAC 开启
  `_compile_loss_cudagraphs`；
- `src/uni_rl/algos/common/learner_boilerplate.py:95`：统一调用 `torch.compile` 并传入
  `triton.cudagraphs` 选项。

这里的顺序很重要：先由 Inductor 融合小 kernel，再 replay 融合后的图。PR #667
的手工方案是在 eager 层捕获，减少了 launch 次数，却保留了较多未融合 kernel。

### 12.3 host 同步碎片

- `src/uni_rl/algos/fast_sac/learner.py:619`：device-side finite optimizer gate；
- `src/uni_rl/algos/fast_sac/learner.py:1319`：多个 scalar metric 合并为一次 D2H；
- `src/uni_rl/algos/fast_sac/learner.py:1328`：actor metrics 延迟到 learner cycle 尾部；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1298` 附近：8 critic + 2 actor 的实际循环；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1392` 附近：读取延迟 metrics；
- `src/uni_rl/offpolicy/double_buffer_runner.py:662` 附近：inference CUDA event 计时，避免
  多余的显式 synchronize。

`.item()`、`if torch.isfinite(cuda_tensor)` 和显式 `synchronize()` 都可能迫使 CPU
等待 GPU。单次看很小，放在每个 update 中就会把一个连续 burst 切成很多碎片。

### 12.4 target network 更新和手工 Graph 安全性

- `src/uni_rl/algos/common/learner_boilerplate.py:23`：Polyak 更新改为单次
  `_foreach_lerp_`；
- `src/uni_rl/algos/fast_sac/learner.py:810` 附近：critic capture candidate 可捕获
  target update；
- `src/uni_rl/algos/fast_sac/learner.py:1581`：声明 target update 是否已在 Graph 中；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1298` 附近：若已捕获则不再外部重复更新。

### 12.5 配置所有权

以下路径相对于 `UniLab` 仓库：

- `src/unilab/conf/sac/config.yaml:14`：batch size 8192；
- `src/unilab/conf/sac/config.yaml:18`：policy frequency 4；
- `src/unilab/conf/sac/config.yaml:25`：actor/critic 网络宽度；
- `src/unilab/conf/sac/config.yaml:37`：默认 `use_compile=true`；
- `src/unilab/conf/sac/config.yaml:38-41`：四个手工 Graph 开关默认 false；
- `src/unilab/conf/sac/task/g1_walk_flat/mujoco.yaml:12`：2048 environments；
- `src/unilab/conf/sac/task/g1_walk_flat/mujoco.yaml:16`：8 updates per step。

### 12.6 对应测试

- `tests/algos/test_fast_sac_compile.py:73`：确认 critic/actor compiled hot paths 使用
  Inductor CUDA Graph；
- `tests/algos/test_fast_sac_compile.py:582`：actor update 不积累无用 critic 权重梯度；
- `tests/algos/test_fast_sac_compile.py:599`：非有限 loss/gradient 会跳过 optimizer step；
- `tests/algos/test_offpolicy_runner_unit.py:527`：target update 捕获前后的 runner 行为；
- `tests/algos/test_offpolicy_runner_unit.py:632`：runtime manifest 能准确说明实际 Graph 路径。

## 13. 整体推理链

1. 早期问题确实是 CPU launch 太多，CUDA Graph 的方向合理。
2. 当前版本中手工 Graph 结构仍在，并未被简单删除或完全破坏。
3. 但 4090 上 `torch.compile` 会融合许多小 kernel；手工 Graph 捕获未融合 eager
   kernels，实测约 29 ms，不能因为名字叫 CUDA Graph 就默认更快。
4. 基线 actor 更新还会为 critic 参数计算无用的权重梯度；最终版把冻结边界放在
   compiled callable 外，减少这部分反向传播开销，同时保留 `dQ/da`。
5. 排查中若把冻结动作误放进 compiled loss，会出现 graph break；最终版没有这样做。
6. 再去掉 loss finite check、metrics `.item()`、inference sync 等 host 等待点，完整
   learner burst 恢复连续。
7. `max_autotune` 虽把微基准降到约 15.63 ms，但首次编译多约 35 秒，并让 collector
   在 tick 11 超时，因此被明确拒绝。
8. 最终三次完整训练 tail-150 mean 均低于 20 ms，说明优化不仅在孤立微基准中
   有效，也能落到真实 G1/MuJoCo 训练路径。

## 14. 可直接用于汇报的摘要

> 本次排查确认，当前 SAC 性能回退并非 PR #667 的手工 CUDA Graph 被直接删除。
> 基线真实默认路径是 `torch.compile`，但明确关闭 Inductor cudagraph，且手工 Graph
> 四个开关也都为 false；在 RTX 4090 上手工 Graph 捕获未融合 eager kernels，微基准
> 约为 29 ms。最终版开启“Inductor 先融合、再 cudagraph replay”，并把 critic 参数
> 冻结放在 compiled callable 外，避免基线为 critic 计算无用权重梯度；同时通过
> device-side finite gate、批量 metrics D2H、移除冗余 inference sync 等方式减少
> host 同步碎片。排查中曾把冻结动作放进 compiled loss，产生 graph break，但那是
> 中间实验写法，不是 `77450d2` 的源代码。最终在 RTX 4090、2048 env、batch 8192、
> 每周期 8 critic + 2 actor 的完整 G1/MuJoCo 训练中，三次 300-iteration 实验的
> tail-150 learner mean 分别为 19.177、19.140、18.749 ms，平均 19.022 ms，相比
> 修复前约 24.56 ms 降低约 22.6%，达到单卡不超过 20 ms 的验收目标。
