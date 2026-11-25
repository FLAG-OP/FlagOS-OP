# 性能回归对比: biren
生成时间: 2026-11-24T04:56:14.159266+00:00

- 阈值: WARN ≥ 20% · FAIL ≥ 30%（仅对 latency_ms 门禁）
- 基线: perf/baselines/biren.json（commit `d93d6d65e8ee`，更新于 2026-11-24T04:55:47）
- 环境: 一致（仅提示，不阻断）

| case | 基线 ms | 本次 ms | Δ | 判定 | 附加指标 |
|---|---|---|---|---|---|
| ops.matmul.batch_broadcast.bfloat16.backward.torch | 5.006 | 5.006 | +0.0% | **OK** | GFLOPS=0.483 max_check_error=0.007 |
| ops.matmul.batch_broadcast.bfloat16.forward.torch | 2.420 | 2.420 | +0.0% | **OK** | GFLOPS=0.5 max_check_error=0.002 |
| ops.matmul.batch_broadcast.float32.backward.torch | 5.471 | 5.471 | +0.0% | **OK** | GFLOPS=0.442 max_check_error=0.0 |
| ops.matmul.batch_broadcast.float32.forward.torch | 2.690 | 2.690 | +0.0% | **OK** | GFLOPS=0.45 max_check_error=0.0 |
| ops.matmul.batch_tail.bfloat16.backward.torch | 1.823 | 1.823 | +0.0% | **OK** | GFLOPS=0.442 max_check_error=0.002 |
| ops.matmul.batch_tail.bfloat16.forward.torch | 0.853 | 0.853 | +0.0% | **OK** | GFLOPS=0.473 max_check_error=0.001 |
| ops.matmul.batch_tail.float32.backward.torch | 1.888 | 1.888 | +0.0% | **OK** | GFLOPS=0.427 max_check_error=0.0 |
| ops.matmul.batch_tail.float32.forward.torch | 0.897 | 0.897 | +0.0% | **OK** | GFLOPS=0.45 max_check_error=0.0 |
| ops.matmul.dot.bfloat16.backward.torch | 0.730 | 0.730 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.dot.bfloat16.forward.torch | 0.283 | 0.283 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.dot.float32.backward.torch | 0.729 | 0.729 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.dot.float32.forward.torch | 0.298 | 0.298 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.large_input.bfloat16.backward.torch | 0.772 | 0.772 | +0.0% | **OK** | GFLOPS=0.116 max_check_error=0.002 |
| ops.matmul.large_input.bfloat16.forward.torch | 0.348 | 0.348 | +0.0% | **OK** | GFLOPS=0.128 max_check_error=0.001 |
| ops.matmul.large_input.float32.backward.torch | 0.803 | 0.803 | +0.0% | **OK** | GFLOPS=0.111 max_check_error=0.0 |
| ops.matmul.large_input.float32.forward.torch | 0.424 | 0.424 | +0.0% | **OK** | GFLOPS=0.105 max_check_error=0.0 |
| ops.matmul.matrix_regular.bfloat16.backward.torch | 0.790 | 0.790 | +0.0% | **OK** | GFLOPS=31.868 max_check_error=0.002 |
| ops.matmul.matrix_regular.bfloat16.forward.torch | 0.352 | 0.352 | +0.0% | **OK** | GFLOPS=35.713 max_check_error=0.002 |
| ops.matmul.matrix_regular.float32.backward.torch | 0.782 | 0.782 | +0.0% | **OK** | GFLOPS=32.199 max_check_error=0.0 |
| ops.matmul.matrix_regular.float32.forward.torch | 0.352 | 0.352 | +0.0% | **OK** | GFLOPS=35.764 max_check_error=0.0 |
| ops.matmul.matrix_tail.bfloat16.backward.torch | 0.777 | 0.777 | +0.0% | **OK** | GFLOPS=2.678 max_check_error=0.002 |
| ops.matmul.matrix_tail.bfloat16.forward.torch | 0.350 | 0.350 | +0.0% | **OK** | GFLOPS=2.969 max_check_error=0.001 |
| ops.matmul.matrix_tail.float32.backward.torch | 0.809 | 0.809 | +0.0% | **OK** | GFLOPS=2.572 max_check_error=0.0 |
| ops.matmul.matrix_tail.float32.forward.torch | 0.360 | 0.360 | +0.0% | **OK** | GFLOPS=2.889 max_check_error=0.0 |
| ops.matmul.matrix_tiny.bfloat16.backward.torch | 0.747 | 0.747 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.matrix_tiny.bfloat16.forward.torch | 0.343 | 0.343 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.matrix_tiny.float32.backward.torch | 0.800 | 0.800 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.matrix_tiny.float32.forward.torch | 0.361 | 0.361 | +0.0% | **OK** | GFLOPS=0.0 max_check_error=0.0 |
| ops.matmul.matrix_vector.bfloat16.backward.torch | 0.747 | 0.747 | +0.0% | **OK** | GFLOPS=0.013 max_check_error=0.002 |
| ops.matmul.matrix_vector.bfloat16.forward.torch | 0.389 | 0.389 | +0.0% | **OK** | GFLOPS=0.012 max_check_error=0.0 |
| ops.matmul.matrix_vector.float32.backward.torch | 0.817 | 0.817 | +0.0% | **OK** | GFLOPS=0.012 max_check_error=0.0 |
| ops.matmul.matrix_vector.float32.forward.torch | 0.365 | 0.365 | +0.0% | **OK** | GFLOPS=0.013 max_check_error=0.0 |
| ops.matmul.noncontiguous.bfloat16.backward.torch | 0.733 | 0.733 | +0.0% | **OK** | GFLOPS=0.381 max_check_error=0.002 |
| ops.matmul.noncontiguous.bfloat16.forward.torch | 0.386 | 0.386 | +0.0% | **OK** | GFLOPS=0.362 max_check_error=0.001 |
| ops.matmul.noncontiguous.float32.backward.torch | 0.756 | 0.756 | +0.0% | **OK** | GFLOPS=0.369 max_check_error=0.0 |
| ops.matmul.noncontiguous.float32.forward.torch | 0.399 | 0.399 | +0.0% | **OK** | GFLOPS=0.349 max_check_error=0.0 |
| ops.matmul.rank_one.bfloat16.backward.torch | 0.683 | 0.683 | +0.0% | **OK** | GFLOPS=0.014 max_check_error=0.0 |
| ops.matmul.rank_one.bfloat16.forward.torch | 0.346 | 0.346 | +0.0% | **OK** | GFLOPS=0.014 max_check_error=0.001 |
| ops.matmul.rank_one.float32.backward.torch | 0.715 | 0.715 | +0.0% | **OK** | GFLOPS=0.013 max_check_error=0.0 |
| ops.matmul.rank_one.float32.forward.torch | 0.363 | 0.363 | +0.0% | **OK** | GFLOPS=0.013 max_check_error=0.0 |
| ops.matmul.vector_matrix.bfloat16.backward.torch | 0.740 | 0.740 | +0.0% | **OK** | GFLOPS=0.01 max_check_error=0.002 |
| ops.matmul.vector_matrix.bfloat16.forward.torch | 0.349 | 0.349 | +0.0% | **OK** | GFLOPS=0.011 max_check_error=0.0 |
| ops.matmul.vector_matrix.float32.backward.torch | 0.759 | 0.759 | +0.0% | **OK** | GFLOPS=0.01 max_check_error=0.0 |
| ops.matmul.vector_matrix.float32.forward.torch | 0.369 | 0.369 | +0.0% | **OK** | GFLOPS=0.01 max_check_error=0.0 |
| ops.matmul.zero_input.bfloat16.backward.torch | 0.772 | 0.772 | +0.0% | **OK** | GFLOPS=0.006 max_check_error=0.0 |
| ops.matmul.zero_input.bfloat16.forward.torch | 0.345 | 0.345 | +0.0% | **OK** | GFLOPS=0.006 max_check_error=0.0 |
| ops.matmul.zero_input.float32.backward.torch | 0.796 | 0.796 | +0.0% | **OK** | GFLOPS=0.005 max_check_error=0.0 |
| ops.matmul.zero_input.float32.forward.torch | 0.362 | 0.362 | +0.0% | **OK** | GFLOPS=0.006 max_check_error=0.0 |

## 结论: OK
FAIL 0 · WARN 0 · NEW 0 · 共 48 条