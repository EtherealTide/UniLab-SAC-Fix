# UniLab FastSAC 单卡性能优化报告

## 结论

在相同真实训练 workload 下，最新版 `unilab_rl`（`f42570d`）的 learner 稳态均值为
`27.168 ± 0.377 ms`；应用本仓库 patch 后为 `18.289 ± 0.217 ms`。绝对下降
`8.879 ms`，相对下降 `32.68%`，等价 update 吞吐提升 `48.55%`。三次 final
运行的均值全部满足 mentor 的 `<=20 ms` 指标；p90 仍约 `21.9--22.8 ms`，所以
当前承诺是“均值达标”，不是“每次或 p90 均低于 20 ms”。

## 版本关系：三个容易混淆的路径

1. **手工 CUDA Graph（PR #667）**：显式打开 `use_cuda_graph_critic/actor`，直接捕获
   eager 更新。诊断微基准中 `graph_packed` 约 29 ms；它捕获了较多未融合的小 kernel，
   因而不是当前 4090 上最优的生产路径。
2. **当前 upstream baseline（`f42570d`）**：生产配置为 `use_compile=true`，四个手工
   Graph 开关全部为 false；公共 compile owner 又把 `triton.cudagraphs` 硬编码为
   false。因此这是“Inductor 编译/融合，但不 replay CUDA Graph”的路径，也是本报告
   的正式 baseline。
3. **改进版（`f42570d` + patch）**：仍保持四个手工 Graph 开关为 false，但 FastSAC
   允许 `torch.compile` 的 Inductor CUDA Graph replay；同时清理热路径中的同步、无用
   梯度和重复 target-update launch。这不是把手工 Graph 开关重新打开。

## 实验设置与正式结果

硬件：RTX 4090；任务：G1WalkFlat/MuJoCo；2048 envs；batch 8192；每周期 8 次 critic
update；`policy_frequency=4`（2 次 actor update）；AMP；300 iterations。指标读取
`timing/learner_train_ms` 的 tail 150（iterations 151--300）。

| 版本 | run | mean (ms) | median | p90 | p95 | mean ≤20 |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| upstream `f42570d` | 1 | 26.806 | 26.550 | 30.304 | 31.354 | 否 |
| upstream `f42570d` | 2 | 27.558 | 27.493 | 30.708 | 32.567 | 否 |
| upstream `f42570d` | 3 | 27.141 | 27.023 | 30.322 | 31.711 | 否 |
| `f42570d` + patch | 1 | 18.475 | 18.576 | 22.753 | 23.618 | 是 |
| `f42570d` + patch | 2 | 18.341 | 18.825 | 22.175 | 23.134 | 是 |
| `f42570d` + patch | 3 | 18.051 | 18.670 | 21.897 | 22.635 | 是 |

三次 run 汇总：

| 指标 | baseline | final |
| --- | ---: | ---: |
| mean 的平均 ± 样本 SD | 27.168 ± 0.377 ms | 18.289 ± 0.217 ms |
| median 的平均 | 27.022 ms | 18.691 ms |
| p90 的平均 | 30.444 ms | 22.275 ms |
| p95 的平均 | 31.877 ms | 23.129 ms |

原始统计 JSON 保存在 `results/latest_main_real_training_results.json`；旧的
`results/mentor_real_training_results.json` 保留为历史数据，不能用于本次结论。

## 为什么会回退

- 公共 `LearnerBoilerplateMixin` 对所有 compile loss 固定使用
  `options={"triton.cudagraphs": False}`，所以生产 `torch.compile` 没有 replay 图。
- critic/actor 更新内反复执行 `torch.isfinite(loss)` 等 host-visible 分支，Python
  必须等待 GPU 才能决定是否 step，8 次 critic + 2 次 actor 把 GPU burst 切碎。
- 每次 update 都 `.item()` 多个 metric，产生重复 D2H/同步。
- actor 需要通过 critic 得到 `dQ/da`，但 baseline 没有冻结 critic 参数，额外计算了
  不会被 actor optimizer 使用的 `dQ/dW`。
- 手工 Graph 路径中 target-network Polyak update 在 graph 外部，且 inference 计时有
  一次多余的 stream synchronize。这些边界会增加 launch 和等待。

需要特别澄清：baseline 中没有在 compiled loss 内执行
`for parameter in self.qnet.parameters(): parameter.requires_grad_(False)`。那是排查时
尝试过的中间实验，不是 upstream 回退原因。

## 改进措施与代码位置

- `src/uni_rl/algos/common/learner_boilerplate.py:54,100`：增加
  `_compile_loss_cudagraphs` 开关，将其传给 Inductor `triton.cudagraphs`。
- `src/uni_rl/algos/fast_sac/learner.py:399`：FastSAC 开启 Inductor cudagraph；
  `:619`：device-side finite optimizer gate；`:656`：compiled callable 外冻结
  critic 参数；`:1319`：一次性批量读取 metric；`:1505`：actor 更新使用冻结上下文。
- `src/uni_rl/offpolicy/double_buffer_runner.py:254`：runtime manifest 记录实际路径
  和 fallback；`:681`：CUDA Event inference 计时；`:1313--1411`：延迟并在周期末
  读取 critic/actor metrics，且在 graph 已捕获 target update 时跳过外部更新。
- `src/unilab/conf/sac/config.yaml:37--41`：生产配置中的 compile/手工 Graph 开关；
  `src/unilab/conf/sac/task/g1_walk_flat/mujoco.yaml:12,16`：2048 envs 和 8 updates。

## Profiler 证据

当前机器未安装 Nsight Systems（`nsys` 不可用），因此没有虚构 kernel/API 级别的
`cudaGraphLaunch` 统计。已有证据是 UniLab TraceRecorder 在真实 G1/MuJoCo 训练中
记录的应用级 Perfetto/Chrome trace + CUDA Events：

| 真实训练事件（稳态 median） | 旧 trace baseline | 旧 trace final | 变化 |
| --- | ---: | ---: | ---: |
| learner/update_phase | 27.356 ms | 18.931 ms | -30.80% |
| learner/update_critic | 2.187 ms | 1.373 ms | -37.22% |
| learner/update_actor | 2.525 ms | 0.934 ms | -63.01% |
| learner/replay_sample | 1.179 ms | 1.223 ms | +3.73% |
| GPU replay gather | 0.302 ms | 0.280 ms | -7.28% |

这组 trace 来自较早的 `77450d2 → c543650` 对照（文件见
`/tmp/sac_report_profile_77450d2/` 和 `/tmp/sac_report_profile_c543650/`），仅用于
解释收益发生在哪里；本报告正式验收数字全部来自最新 `f42570d` A/B。它显示 replay
和 gather 基本不变，而 critic/actor 热路径显著缩短，与上述代码机制一致。

## 质量验证与限制

最新版 upstream + patch 已在 preflight worktree 完成：focused tests `59 passed,
9 skipped`；Ruff check、Ruff format check、Mypy 均通过；完整 pytest 为 `324 passed,
36 skipped, 3 deselected`，另有 2 个测试因受限环境无法解析 `127.0.0.1` 的 Gloo
初始化失败，非代码断言失败。正式 PR 应在正常网络/本机环境再跑 `make check` 和
`make test-all`。

本报告未把早期 `77450d2 → c543650` 的均值写入正式结论，也未把手工 Graph 的诊断
微基准当作生产 baseline。若验收标准改成 p90 ≤20 ms，当前结果仍需进一步优化。
