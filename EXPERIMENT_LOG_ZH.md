# 从初始代码到最终优化：UniLab SAC 性能实验日志

这份文档回答的是“每一步为什么做、改了什么、结果如何、下一步怎么决定”，
而不是只复现最终通过的数字。

## 0. 固定实验口径

所有阶段尽量保持下面的 workload 不变：

```text
GPU              RTX 4090
obs / critic / action   98 / 101 / 29
batch size        8192
updates per step  8
policy frequency  4       # 8 次 critic 中有 2 次 actor
actor / critic hidden    512 / 768
num_atoms         101
AMP               BF16
```

微基准的“真实训练近似”还要加：

```text
warmup=10 或 20
iterations=50 或 100
inter-cycle-sleep-ms=32
read-production-metrics=true
input-layout=replay_packed_views
```

为什么必须固定这些参数：如果同时改变 batch、网络宽度、GPU 时钟状态和 replay
输入布局，就无法知道性能变化来自哪一项。

## 1. 建立初始基线：先不要改代码

### 1.1 准备初始版本

`unilab_rl` 的初始性能版本是 `77450d2`。在一个临时 worktree 中操作，避免
影响已经修好的工作区：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl
git worktree add /tmp/unilab-rl-sac-baseline 77450d2
```

检查 Python 是否加载临时版本：

```bash
cd /home/pc823/桌面/unilabsim/UniLab
PYTHONPATH=/tmp/unilab-rl-sac-baseline/src:/home/pc823/桌面/unilabsim/unisim/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python -c "import uni_rl; print(uni_rl.__file__)"
```

输出必须指向 `/tmp/unilab-rl-sac-baseline/src/uni_rl`。如果指向 site-packages，
后面的实验全部无效。

### 1.2 基线完整训练

初始版本先跑 150 iterations，主要目的不是最终验收，而是确认 mentor 说的
“约 24 ms 回退”确实存在。完整训练命令见
[REPRODUCTION_REPORT_ZH.md](REPRODUCTION_REPORT_ZH.md) 第 8 节，只需把：

```text
PYTHONPATH 改成 /tmp/unilab-rl-sac-baseline/src:...
training.log_dir 改成 /tmp/sac-stage-00-baseline
algo.max_iterations 改成 150
```

然后统计：

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --offline \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/analyze_tensorboard.py \
  /tmp/sac-stage-00-baseline --tail 75
```

历史基线后半段结果约为：

```text
mean     24.56 ms
median   24.47 ms
p90      27.59 ms
```

这一步的判断：问题可以稳定复现，不是一次偶然的 GPU 抖动。接下来才进入
微基准定位，避免每次修改都等待完整训练。

## 2. 第一轮定位：手工 CUDA Graph 是否比 compile 更快？

### 2.1 运行统一微基准

在初始版本上运行：

```bash
cd /home/pc823/桌面/unilabsim/unilab_rl
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/unilab-rl-sac-baseline/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/bench_fast_sac.py \
  --cases compile,graph_packed \
  --warmup 5 --iterations 30 \
  --unilab-rl-root /tmp/unilab-rl-sac-baseline \
  --output /tmp/sac-stage-01-compile-vs-manual.json
```

查看：

```bash
jq '.results[] | {case:.case.name, host:.cycle_host_ms}' \
  /tmp/sac-stage-01-compile-vs-manual.json
```

历史结果：

| 路径 | median |
| --- | ---: |
| `torch.compile` | 约 19.4 ms |
| 手工 Graph + packed staging | 约 29.1 ms |

### 2.2 这一步得到的结论

手工 Graph 没有“失效”到完全不执行；它确实在 replay。但在 RTX 4090 上，
它捕获的是较多未融合的 eager kernel，反而慢。因此不能简单地把 PR #667 的
四个开关重新打开作为最终方案。

相关代码：

- 手工 Graph 入口：
  `unilab_rl/src/uni_rl/algos/fast_sac/learner.py` 中
  `update_critic_cuda_graph`、`update_actor_cuda_graph`；
- 配置开关：
  `UniLab/src/unilab/conf/sac/config.yaml:37-41`。

## 3. 第二轮：确认 Inductor CUDA Graph 是否有效

### 3.1 A/B 实验

这一步比较普通 compile 和强制关闭/打开 Inductor cudagraph：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/unilab-rl-sac-baseline/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/bench_fast_sac.py \
  --cases compile,compile_cudagraphs \
  --warmup 10 --iterations 100 \
  --unilab-rl-root /tmp/unilab-rl-sac-baseline \
  --output /tmp/sac-stage-02-inductor-ab.json
```

对应保存结果也见 `results/compile_inductor_cudagraphs_ab.json`。典型结果为：

```text
普通 compile       median 19.86 ms
Inductor cudagraph median 18.58 ms
```

结论：正确方向是“先由 Inductor 融合，再由 Inductor replay”，而不是只依赖
手工 Graph。

### 3.2 对应源码改动

在 `LearnerBoilerplateMixin._compile_training_methods` 中，将 compile options
设置为：

```python
{"triton.cudagraphs": True}
```

FastSAC 通过：

```python
_compile_loss_cudagraphs = True
```

打开这个路径。最终代码位置：
`unilab_rl/src/uni_rl/algos/fast_sac/learner.py:396-400` 和
`unilab_rl/src/uni_rl/algos/common/learner_boilerplate.py:95`。

## 4. 第三轮：检查 replay layout，而不是先怀疑网络

### 4.1 连续内存和 replay packed views

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/unilab-rl-sac-baseline/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/bench_fast_sac.py \
  --cases compile \
  --input-layout replay_packed_views \
  --warmup 10 --iterations 100 \
  --unilab-rl-root /tmp/unilab-rl-sac-baseline \
  --output /tmp/sac-stage-03-replay-layout.json
```

不加 32 ms sleep 时，历史结果约为 `18.59 ms`；加入真实训练间隔后变为约
`20.83 ms`。这说明纯 replay layout 不是主要回归源，真正的问题更可能是
GPU burst 之间的 host 同步和 clock 状态。

### 4.2 曾尝试但撤回的实验

曾尝试让 replay prepare/pause 与 learner 更新重叠，也尝试直接给 graph 原生
packed layout。没有稳定收益，反而增加复杂度，全部撤回。原则是：没有在同一
workload 下稳定下降，就不进入最终 patch。

## 5. 第四轮：逐个消除 CPU/GPU 同步碎片

这一轮每次只改一项，使用同样的 bursty 命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/pc823/桌面/unilabsim/unilab_rl/src \
UV_CACHE_DIR=/tmp/unilab-uv-cache \
uv run --offline python \
  /home/pc823/桌面/unilabsim/UniLab-SAC-Fix/bench_fast_sac.py \
  --cases compile --input-layout replay_packed_views \
  --warmup 10 --iterations 50 --inter-cycle-sleep-ms 32 \
  --unilab-rl-root /home/pc823/桌面/unilabsim/unilab_rl \
  --output /tmp/sac-stage-N.json
```

每次把输出文件名换成对应阶段；不要覆盖之前的结果。

| 阶段 | 改动 | median | 决策 |
| --- | --- | ---: | --- |
| 初始 burst | 原始 host finite check + 每次 metrics 读取 | 约 20.83 | 需要继续拆分 |
| deferred metrics | metrics 只在周期末读取 | 约 20.36 | 保留方向 |
| no host finite | 去掉 CUDA 上 `torch.isfinite(loss)` 的 host 分支 | 约 17.07 | 收益明显 |
| device finite gate | 用 device scalar + fused AdamW gate | 约 17.29 | 保留，正确性更安全 |
| foreach lerp | Polyak 改为单次 `_foreach_lerp_` | 约 17.64 | 单卡噪声下无收益；只保留为手工 Graph 路径优化 |
| alpha 单独 compile | 给很小的 alpha loss 单独 compile | 约 17.78 | 撤回，复杂度大于收益 |
| actor metrics 延迟 | 只保存最后一次 actor 的 metrics | 约 17.33 | 保留 |
| 最终单图 actor | 修正 actor graph break 后 | 约 16.97 | 保留 |

### 5.1 host finite check 的问题

旧代码在 CUDA tensor 上执行类似：

```python
if torch.isfinite(qf_loss):
```

Python 必须知道条件结果，这会等待 GPU。critic、actor、alpha 每个 update 都有
类似边界，8+2 次更新累积后会把连续 GPU burst 切碎。

最终代码位置：
`unilab_rl/src/uni_rl/algos/fast_sac/learner.py:619-653`。

单卡路径使用 device-side `found_inf`；DP 路径还会在 all-reduce 后检查梯度，
避免某个 rank 的 NaN 被错误写入参数。

### 5.2 metrics 的问题

旧代码对每个 scalar 分别 `.item()`。最终改为把 scalar stack 后一次 D2H，相关
代码在 `learner.py:1319` 附近，runner 的训练循环在
`double_buffer_runner.py:1298` 附近。

这类改动通常不会改变 loss 数值，只减少 CPU 等 GPU 的次数。因此每一步都应
同时检查训练是否仍能完成，而不是只看毫秒数。

## 6. 第五轮：定位真正的 graph break

### 6.1 失败尝试：在 compiled loss 内冻结 critic

为了避免 actor 更新时累计无用 critic 权重梯度，最初把下面逻辑写进
`_actor_loss_tensors`：

```python
for parameter in self.qnet.parameters():
    parameter.requires_grad_(False)
q_outputs = self.qnet(critic_obs, actions)
for parameter in self.qnet.parameters():
    parameter.requires_grad_(True)
```

这看起来合理，但它修改的是 Tensor autograd metadata，不是普通张量计算。
用 Dynamo explain 检查后发现：

```text
3 graphs / 2 graph breaks
```

这正是新版性能回退的关键原因。它不能只看最终 loss 是否正确，必须检查图结构。

### 6.2 正确修复

把冻结动作放在 compiled callable 外面：

```python
with self._critic_parameters_frozen():
    actor_loss, policy_entropy, action_std = self._actor_loss_tensors(...)
```

compiled callable 内恢复为纯张量计算。这样得到：

```text
1 graph / 0 graph breaks
```

同时仍然满足：

- critic 参数不累计 actor backward 的无用梯度；
- action 仍保留 `dQ/da`，actor 仍可学习。

对应源码：

- context manager：`learner.py:656`；
- actor graph capture：`learner.py:767`；
- 普通 actor update：`learner.py:1518`；
- compiled loss 本体：`learner.py:736`。

## 7. 第六轮：target update 和 manual Graph 边界

手工 critic Graph 每次 replay 后都需要 Polyak target update。旧路径把它放在
Graph 外部，导致每次 critic 之间多出 graph-external kernel。

最终做法：

1. capture candidate 支持 `update_target=True`；
2. Graph replay 内捕获 Polyak 更新；
3. runner 通过 `cuda_graph_critic_captures_target_update` 判断是否还需要外部
   `soft_update_target()`。

相关位置：

- `learner.py:800` 附近的 `_update_critic_capture_candidate`；
- `learner.py:1581` 的属性；
- `double_buffer_runner.py:1370` 附近的判断。

这项优化对默认 Inductor 路径不是主要收益，但可以防止手工 Graph 选项在未来
出现重复 target update。

## 8. 第七轮：max_autotune 为什么没有进入最终方案

实验方式是在 compile options 中额外加入：

```python
{"max_autotune": True}
```

微基准结果约为：

```text
普通最终 bursty       17.33 ms
max_autotune           15.63 ms
```

但首次 Triton autotune 多花约 35 秒。实际 G1 训练中 collector 在 tick 11 等待
inference 超时并退出，训练没有完成。对默认 5000 iterations 粗略计算，省下的
每周期 1.7 ms 也不足以抵消冷启动成本。

所以这项实验的结论不是“速度不快”，而是“端到端不划算且会破坏启动可靠性”，
最终明确撤回。结果保存在 `results/bursty_max_autotune.json`。

## 9. 第八轮：真实完整训练验收

所有微基准通过后，才运行真实命令。每次必须使用新的 log_dir：

```text
/tmp/unilab-sac-fix-single-graph-1
/tmp/unilab-sac-fix-single-graph-2
/tmp/unilab-sac-fix-single-graph-3
```

命令见 [REPRODUCTION_REPORT_ZH.md](REPRODUCTION_REPORT_ZH.md) 第 8 节。

运行摘要必须满足：

```json
{
  "status": "completed",
  "completed_iterations": 300,
  "runtime_manifest.cuda_graph.inductor_critic_cudagraphs": true,
  "runtime_manifest.cuda_graph.inductor_actor_cudagraphs": true,
  "runtime_manifest.cuda_graph.device_finite_optimizer_gating": true,
  "runtime_manifest.cuda_graph.fallback_reasons": []
}
```

三次 tail-150 结果：

| run | mean | median | p90 |
| --- | ---: | ---: | ---: |
| 1 | 19.177 ms | 18.941 ms | 23.857 ms |
| 2 | 19.140 ms | 19.416 ms | 22.428 ms |
| 3 | 18.749 ms | 18.961 ms | 22.556 ms |

三次平均为 `19.022 ms`。这一步才是最终验收；微基准只能说明 learner 本体有
改善，不能替代真实 collector + replay + learner 组合。

## 10. 每个阶段如何决定“继续还是撤回”

使用下面的判断顺序：

1. 先确认 workload 和代码版本一致；
2. 再确认训练是否完成，失败的实验不能按“速度很快”算成功；
3. 比较 median 和 p90，不只看单次 minimum；
4. 做正确性测试：finite、NaN、RNG、参数更新；
5. 只有稳定收益才保留；一次变快但引入新 graph tree 或启动延迟就撤回；
6. 最终必须回到 300-iteration 真实训练验证。

## 11. 最终思路总结

```text
初始完整训练约 24.56 ms
        │
        ├─ 手工 Graph A/B：约 29 ms，发现“Graph ≠ 自动更快"
        │
        ├─ Inductor cudagraph：约 18.6–19 ms，确定主路线
        │
        ├─ replay layout：不是主因
        │
        ├─ host finite + 多次 .item()：造成同步碎片
        │       └─ device gate + 批量 D2H，微基准降到约 17 ms
        │
        ├─ actor 内 requires_grad_：3 graphs / 2 breaks
        │       └─ 移到 compiled callable 外，恢复 1 graph / 0 break
        │
        ├─ max_autotune：微基准快，但冷启动导致真实训练超时，撤回
        │
        └─ 三次完整训练：19.177 / 19.140 / 18.749 ms，最终通过
```

这说明 mentor 关于“新代码可能破坏 CUDA Graph 结构”的方向是对的，但具体
表现不是手工 Graph 被删除，而是 compiled actor loss 的 graph break 加上多处
host synchronization，使原本连续的 GPU 工作重新变得碎片化。
