# UniLab SAC 单卡训练性能优化报告

## 1. 摘要

本次工作针对 UniLab 的 G1/MuJoCo SAC 单卡训练回退进行定位和优化。验收指标为
TensorBoard 的 `timing/learner_train_ms`：每次真实训练运行 300 iterations，剔除
前 150 次启动、编译和预热数据，对第 151--300 次取算术平均，要求不超过 20 ms。

在 RTX 4090 上，修复前 `unilab_rl` commit `77450d2` 的三次独立真实训练均值为
`27.234 ± 0.358 ms`；修复后 commit `c543650` 为 `18.050 ± 0.504 ms`。learner
延迟降低 `9.184 ms`，相对降低 `33.72%`，等价 learner update 吞吐提升约
`50.88%`。修复后三次运行均满足 `mean <= 20 ms`。

除 TensorBoard 外，本报告使用真实 G1/MuJoCo 训练中的 Perfetto/Chrome trace 和
CUDA Event 做应用级 profiling。50-iteration 对称 trace（剔除前 5 个编译/启动
cycle）显示，learner update phase 的 median 从 `27.356 ms` 降至 `18.931 ms`；
actor update median 从 `2.525 ms` 降至 `0.934 ms`。replay sample 与 replay GPU
gather 基本不变，说明主要收益发生在 learner 的 critic/actor 热路径，而不是环境或
replay 偶然加速。

## 2. 版本与实验环境

| 项目 | baseline | final |
| --- | --- | --- |
| `unilab_rl` commit | `77450d2ad00d1c56c320e45a2c4150dec2077acc` | `c54365034f2ebbb2d419bc03f73bd47e2e5304ae` |
| 实际 import | `/tmp/unilab-rl-sac-baseline/src/uni_rl/__init__.py` | `/home/pc823/桌面/unilabsim/unilab_rl/src/uni_rl/__init__.py` |
| UniLab commit | `36fb680d1f56931a46783be6af7aa37eb18fdbd2`（dirty） | 相同工作区 |

硬件和 workload：

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，49140 MiB |
| CPU | AMD Ryzen 9 9950X 16-Core Processor |
| 仿真任务 | G1WalkFlat / MuJoCo |
| seed | 1 |
| environments | 2048 |
| batch size | 8192 |
| SAC update | 每周期 8 critic；`policy_frequency=4`，即 2 actor |
| 网络 | actor hidden 512；critic hidden 768；C51 atoms 101 |
| 数值精度 | AMP，`amp_dtype=auto`（本机为 BF16） |
| Graph 配置 | `use_compile=true`；四个手工 `use_cuda_graph_*` 开关均为 false |

六次正式运行均完成 300 iterations，每个 log directory 只有一个 TensorBoard event
文件，避免了多次运行数据混入同一目录。完整配置保存在各目录的 `run_config.json`，
完成状态和实际 runtime manifest 保存在 `run_summary.json`。

## 3. 原始代码实际运行路径

原始生产配置并不使用 PR #667 的手工 CUDA Graph。UniLab 配置为：

```yaml
use_compile: true
use_cuda_graph_critic: false
use_cuda_graph_actor: false
use_cuda_graph_critic_packed_staging: false
use_cuda_graph_actor_packed_staging: false
```

基线 `LearnerBoilerplateMixin._compile_training_methods()` 调用 `torch.compile`，但明确
传入：

```python
{"options": {"triton.cudagraphs": False}}
```

因此基线生产路径是“Inductor 编译/融合，但不使用 Inductor CUDA Graph replay”。
源码中虽然存在 `update_critic_cuda_graph()` 和 `update_actor_cuda_graph()`，只有显式
打开手工 Graph 开关时 runner 才会调用它们。

诊断 benchmark 曾显式打开这些开关，得到手工 `graph_packed` 约 29 ms，而普通
compile 约 19--20 ms。该结果仅用于选择优化路线，不是生产 baseline：在当前
PyTorch 2.8/RTX 4090 上，手工 Graph 捕获较多未融合 eager kernels，反而不如先由
Inductor 融合、再 replay 融合结果。

## 4. 修复内容及性能机制

### 4.1 开启 Inductor CUDA Graph

最终代码在 FastSAC 中设置：

```python
_compile_loss_cudagraphs = True
```

公共 compile owner 根据该属性设置 `triton.cudagraphs`。默认生产路径变成：

```text
torch.compile/Inductor 融合小 kernel
        ↓
CUDA Graph replay 融合后的 loss 计算
```

它与 PR #667 的手工 eager Graph 是两条不同路径。最终正式训练的 runtime manifest
明确记录：

```json
{
  "critic_enabled": false,
  "actor_enabled": false,
  "inductor_critic_cudagraphs": true,
  "inductor_actor_cudagraphs": true,
  "device_finite_optimizer_gating": true,
  "fallback_reasons": []
}
```

即手工 Graph 未开启，critic/actor 的 Inductor CUDA Graph 正常生效，且没有 fallback。

### 4.2 移除热路径中的 host synchronization

基线每次 critic/actor update 都存在 host-visible 操作：

```python
if torch.isfinite(loss):
    ...

loss.item()
metric.item()
```

Python 为执行分支或取得标量，必须等待 CUDA 完成。在一个周期内这些操作重复出现在
8 次 critic 和 2 次 actor 更新中，会把连续 GPU burst 切成多段。

最终版使用 device-side finite optimizer gate，在 GPU 上设置 fused optimizer 的
`found_inf`；metrics 只在周期末读取，并把多个 scalar 合成一次 D2H。actor metric
在 GPU 上先形成稳定副本，避免下一次 CUDA Graph replay 覆盖前一次输出。

受控单项实验中，延迟 metrics 后 median 约为 `20.36 ms`；去掉 host finite 分支后
降至约 `17.07 ms`。这些是排查工作区中的单项开关实验，不是独立 Git 版本，但说明
host synchronization 是主要可操作瓶颈。

### 4.3 避免 actor backward 计算无用 critic 权重梯度

基线 actor loss 需要通过 critic 计算 `Q(s,a)`，但没有冻结 critic 权重。于是 actor
backward 除了必需的 `dQ/da`，还会计算 critic 参数的 `dQ/dW`，而 actor optimizer
根本不会更新这些 critic 参数。

最终版新增 `_critic_parameters_frozen()`，并在调用 compiled actor loss 的外部冻结
critic 参数：

```python
with self._critic_parameters_frozen():
    actor_loss, policy_entropy, action_std = self._actor_loss_tensors(...)
```

这样保留 `dQ/da`，不计算无用 `dQ/dW`。冻结动作放在 compiled callable 外，避免把
`requires_grad_()` 这种 autograd metadata 修改放进 Dynamo 图。

需要澄清：基线 `77450d2` 中不存在“在 compiled loss 内循环执行
`requires_grad_()`”的代码。那是排查过程中尝试过并撤回的中间写法，不能作为基线
回退原因。可验证的基线问题是“完全没有冻结 critic，actor backward 计算无用权重
梯度”。

### 4.4 其他改动

- Polyak target update 从 `_foreach_mul_ + _foreach_add_` 合并为一次
  `_foreach_lerp_`；
- 手工 critic Graph 路径可以捕获 target update，runner 避免重复外部更新；
- inference 使用 CUDA Event 计时，移除仅为计时而产生的额外显式 synchronize；
- runtime manifest 明确记录最终实际 Graph 路径及 fallback 原因。

其中 target capture 主要服务手工 Graph 兼容性，不是本次默认 Inductor 路径的主要
收益来源。

## 5. 正式真实训练结果

正式指标为第 151--300 次 `timing/learner_train_ms`。

| 版本 | run | mean | median | p90 | p95 | `mean <= 20 ms` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | 1 | 26.829 | 26.757 | 30.121 | 30.695 | 否 |
| baseline | 2 | 27.507 | 27.417 | 31.005 | 32.771 | 否 |
| baseline | 3 | 27.366 | 27.280 | 30.473 | 31.448 | 否 |
| final | 1 | 18.318 | 18.290 | 22.680 | 23.425 | 是 |
| final | 2 | 17.469 | 17.071 | 21.527 | 22.360 | 是 |
| final | 3 | 18.364 | 18.368 | 22.020 | 22.993 | 是 |

三次 run 的汇总：

| 指标 | baseline | final | 改善 |
| --- | ---: | ---: | ---: |
| mean 的三次平均 | 27.234 ms | 18.050 ms | -9.184 ms / -33.72% |
| mean 的样本标准差 | 0.358 ms | 0.504 ms | — |
| median 的三次平均 | 27.151 ms | 17.910 ms | -34.04% |
| p90 的三次平均 | 30.533 ms | 22.076 ms | -27.70% |
| p95 的三次平均 | 31.638 ms | 22.926 ms | -27.54% |

以固定 workload 下的 update 频率换算，平均延迟从 27.234 ms 降至 18.050 ms，等价
learner update 吞吐提升：

```text
27.234 / 18.050 - 1 = 50.88%
```

验收目标定义为平均值不超过 20 ms，而不是 p90 不超过 20 ms。最终三次 mean 均
通过；p90 仍为 21.5--22.7 ms。如果 mentor 将验收口径改为 p90 <= 20 ms，则仍需
针对 tail latency 继续优化。

## 6. 真实训练 profiler 结果

为了避免 profiler 开销污染正式 timing，本节另跑 baseline/final 各一次 50-iteration
真实 G1/MuJoCo 训练，打开：

```text
training.trace_enabled=true
training.trace_cuda_events=true
training.nvtx_profile_ranges=true
```

trace 文件：

```text
/tmp/sac_report_profile_77450d2/perfetto_offpolicy_timeline.json
/tmp/sac_report_profile_c543650/perfetto_offpolicy_timeline.json
```

统计按事件时间剔除最早 5 个 learner cycle，排除首次 Inductor 编译和启动长尾。

| 真实训练事件 | baseline median | final median | 变化 |
| --- | ---: | ---: | ---: |
| `learner/update_phase` | 27.356 ms | 18.931 ms | -30.80% |
| `learner/update_critic` | 2.187 ms | 1.373 ms | -37.22% |
| `learner/update_actor` | 2.525 ms | 0.934 ms | -63.01% |
| `learner/replay_sample` | 1.179 ms | 1.223 ms | +3.73% |
| `learner/inference` | 0.526 ms | 0.659 ms | +25.29% |
| GPU replay batch gather | 0.302 ms | 0.280 ms | -7.28% |
| GPU replay storage H2D | 0.159 ms | 0.151 ms | -5.03% |

对应 update phase 的完整稳态统计：

| 版本 | n | mean | median | p90 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 45 | 27.278 | 27.356 | 30.827 | 30.946 |
| final | 45 | 18.577 | 18.931 | 22.377 | 23.365 |

证据指向很清楚：

1. learner update phase 降低约 31%，与无 profiler 的正式 TensorBoard 结果方向一致；
2. actor update 降幅最大，符合“避免 critic 无用权重梯度 + CUDA Graph replay +
   减少 metric 同步”的代码机制；
3. replay sample、GPU gather 和 H2D 没有出现同量级改善，排除了“主要是 replay 或
   环境偶然变快”的解释；
4. inference 略慢但绝对值小，不能解释 learner 总体 9 ms 的下降。

该 trace 是 UniLab TraceRecorder 记录的应用级 CPU slice 和 CUDA Event 粗粒度 GPU
span，不是逐 kernel Nsight Systems 数据。`training.nvtx_profile_ranges=true` 已插入
critic/actor loss、backward、optimizer 等 NVTX 标记，但当前机器未安装 `nsys`，因此
本报告不虚构 `cudaLaunchKernel`/`cudaGraphLaunch` 次数。如果需要逐 kernel/API
证明，可按 `PROFILE_REPRODUCTION_ZH.md` 第 5 节在安装 Nsight Systems 后补采。

## 7. 源代码位置

相对于 `unilab_rl`：

- `src/uni_rl/algos/common/learner_boilerplate.py:95`：统一设置
  `torch.compile(..., options={"triton.cudagraphs": ...})`；
- `src/uni_rl/algos/fast_sac/learner.py:399`：FastSAC 开启 Inductor CUDA Graph；
- `src/uni_rl/algos/fast_sac/learner.py:619`：device-side finite optimizer gate；
- `src/uni_rl/algos/fast_sac/learner.py:656`：critic 参数冻结 context；
- `src/uni_rl/algos/fast_sac/learner.py:1319`：批量读取 metrics；
- `src/uni_rl/algos/fast_sac/learner.py:1505`：普通 actor update；
- `src/uni_rl/offpolicy/double_buffer_runner.py:660`：inference CUDA Event 计时；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1319`：周期内延迟 critic metrics；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1339`：延迟 actor metrics；
- `src/uni_rl/offpolicy/double_buffer_runner.py:1392`：周期末统一读取 actor metrics。

相对于 `UniLab`：

- `src/unilab/conf/sac/config.yaml:37-41`：生产 `torch.compile` 和手工 Graph 开关；
- `src/unilab/conf/sac/config.yaml:85-89`：trace、CUDA Event 和 NVTX 开关；
- `src/unilab/conf/sac/task/g1_walk_flat/mujoco.yaml:12-16`：2048 environments 和
  8 updates per step。

## 8. 正确性与限制

最终修改不改变 SAC loss 定义、网络宽度、batch、update 次数或训练 seed。已有质量
门禁包括 Ruff、Mypy、Pyright、405 passed / 8 skipped 的测试，以及 CUDA finite、
NaN loss/gradient、随机数推进和 Graph replay smoke。

本报告的限制：

1. UniLab 工作区在实验时为 dirty，因此报告同时固定 `unilab_rl` commit、完整
   `run_config.json` 和实际 import 路径；baseline/final 使用同一 UniLab 工作区，
   不影响两者的相对 A/B，但正式归档最好在干净 UniLab commit 上复跑；
2. 当前证据包含真实训练 TensorBoard 和应用级 Perfetto/CUDA Event profiler，但没有
   Nsight Systems 逐 kernel/API 数据；
3. 验收口径是 mean <= 20 ms，不能表述成“所有 iteration 或 p90 都低于 20 ms”；
4. profiler run 只用于解释结构，正式性能数字来自关闭 trace 的六次 300-iteration
   运行。

## 9. 结论

原始回退不是“手工 CUDA Graph 代码被删除”，因为生产 baseline 本来就走
`torch.compile` 且强制关闭 Inductor cudagraph，四个手工 Graph 开关也全部关闭。
最终方案选择“Inductor 先融合、CUDA Graph 再 replay”，并清理每次 update 中的
host finite 分支和 scalar D2H，同步避免 actor backward 计算无用 critic 权重梯度。

三次独立真实训练把 learner tail-150 mean 从 `27.234 ms` 降至 `18.050 ms`，延迟
降低 `33.72%`，三次均通过 `<=20 ms` 验收。真实训练 profiler 同时显示 learner
update phase median 从 `27.356 ms` 降至 `18.931 ms`，主要改善集中在 critic/actor
更新而非 replay。这些结果共同支持代码机制与端到端收益之间的因果解释。
