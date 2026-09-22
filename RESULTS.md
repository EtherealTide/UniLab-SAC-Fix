# Results

## Conclusion

The regression was not caused by the manual CUDA Graph from UniLab PR #667
being structurally broken. On the current RTX 4090 workload, that graph captures
unfused eager kernels and is slower than the default `torch.compile` path:

| Path | Learner cycle median |
| --- | ---: |
| Clean main, `torch.compile` | 19.38 ms |
| Clean main, manual graph + packed staging | 29.14 ms |

The current production path is therefore Inductor fusion plus Inductor CUDA
Graph replay, with the four manual FastSAC graph flags left disabled.

## Root cause

The largest actionable graph regression was in the actor loss. Disabling and
restoring critic parameter gradients inside the function passed to
`torch.compile` caused Dynamo to produce three graphs with two graph breaks.
Moving that metadata operation around the compiled callable restores one graph
with zero graph breaks, while still preserving `dQ/da` and avoiding unused
critic weight gradients.

Other latency came from host-visible finite checks, repeated scalar `.item()`
reads, a redundant inference synchronization, and graph-external target-network
updates on the legacy manual graph path.

## Acceptance result

The primary metric is the arithmetic mean of TensorBoard
`timing/learner_train_ms` over iterations 151–300. Each cycle contains eight
critic and two actor updates with batch size 8192.

| Independent run | Mean | Median | p90 | Result |
| --- | ---: | ---: | ---: | --- |
| 1 | 19.177 ms | 18.941 ms | 23.857 ms | pass |
| 2 | 19.140 ms | 19.416 ms | 22.428 ms | pass |
| 3 | 18.749 ms | 18.961 ms | 22.556 ms | pass |

The mean of the three run means is **19.022 ms**. All three runs satisfy the
mentor target of `<=20 ms`. The earlier full-training baseline was about
24.56 ms, so the final steady-state mean is about 22.6% lower.

## Rejected experiment

Inductor `max_autotune` reduced the isolated bursty microbenchmark to about
15.63 ms, but added roughly 35 seconds of first-use compilation. In a real
training launch, that pause exceeded the collector inference timeout and
aborted at tick 11. It is not part of the final change.

Raw selected results are under `results/`. The three-run summary is
`results/training_repeats.json`. The production change is preserved as a
standard `git am` patch under `patches/`.
