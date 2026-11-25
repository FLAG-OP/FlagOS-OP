# Biren166M matmul evidence

Collected on 2026-11-24 from container `tyanfeng-matmul-supa` with logical
device `supa:0`.

| File | Content |
|---|---|
| `biren_env.log` | device, driver and package versions |
| `biren_accuracy_direct.log` | torch-level direct golden, FP32/BF16, 96 cases |
| `biren_accuracy_a1.log` | A1 golden, FP32/BF16, 96 cases |
| `biren_op_level.log` | A1 interception, three execution modes and CPU isolation |
| `biren_smoke.log` | 7-shape smoke and mode-hit matrix |
| `biren_framework.log` | two-matmul application consumer against a CPU baseline |
| `biren_boundaries.log` | FP16, native vector, Triton and FlagGems boundaries |
| `biren_perf.log` | initial performance run output |
| `biren_dispatch.json` | native, direct torch-level and A1 latency comparison |
| `provenance.json` | measured checkout, runtime and final source hashes |
| `perf/` | archived baseline and per-case raw performance records |

The FP16 probe is deliberately isolated in a child process: the BR10x SDK
terminates the process with `SIGABRT` after reporting
`STATUS_NOT_SUPPORTED`. Triton and FlagGems are reported as unavailable
based on actual `find_spec` results in the measured runtime.

Raw logs are retained without rewriting values. Markdown reports summarize
them and state that the supported-dtype matrix is 96/96, not 144/144.
