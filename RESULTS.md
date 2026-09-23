# SAC performance results

The final mentor-facing analysis is in [MENTOR_REPORT_ZH.md](MENTOR_REPORT_ZH.md).

## Audited real-training result

The comparison uses G1WalkFlat/MuJoCo on one RTX 4090, 2048 environments,
batch size 8192, eight critic updates per cycle, policy frequency four, AMP,
and three independent 300-iteration runs per version. The reported metric is
the arithmetic mean of TensorBoard `timing/learner_train_ms` over iterations
151--300.

| Version | Mean of three run means | Sample SD | Result |
| --- | ---: | ---: | --- |
| Baseline `77450d2` | 27.234 ms | 0.358 ms | fail |
| Final `c543650` | 18.050 ms | 0.504 ms | pass |

Latency decreased by 9.184 ms (33.72%); equivalent learner-update throughput
increased by 50.88%. All three final runs satisfy the `mean <= 20 ms` target.

## Real-training profiler evidence

Separate 50-iteration G1/MuJoCo runs enabled UniLab's Perfetto/Chrome trace
recorder, CUDA Events, and NVTX ranges. After excluding the first five compile
and startup cycles:

| Event median | Baseline | Final |
| --- | ---: | ---: |
| Learner update phase | 27.356 ms | 18.931 ms |
| Critic update | 2.187 ms | 1.373 ms |
| Actor update | 2.525 ms | 0.934 ms |
| Replay sample | 1.179 ms | 1.223 ms |

The improvement is concentrated in learner critic/actor work, while replay is
essentially unchanged. This supports the code-level explanation: Inductor CUDA
Graph replay, removal of repeated host synchronization and scalar reads, and
elimination of unused critic-weight gradients during actor backward.

The current machine does not have Nsight Systems installed. These profiler
results are application-level Perfetto slices and coarse CUDA Event spans, not
invented per-kernel Nsight statistics. See `PROFILE_REPRODUCTION_ZH.md` for the
optional Nsight procedure.

Machine-readable results are in `results/mentor_real_training_results.json`.
