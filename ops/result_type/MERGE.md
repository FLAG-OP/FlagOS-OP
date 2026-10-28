# 多平台合并流程（MERGE）— result_type

> 新平台实现合入本目录的固定流程。核心原则：**黄金 / 测试 / 注册链复用，
> 只写设备相关分支**。

## 七步流程

1. **黄金复用**：`goldendata/` 平台无关（CPU 生成），新平台直接跑
   `script/check_accuracy.py`——不过黄金不谈合并。
2. **实现本体**：Triton 级补 `kernel/triton_level.py` 的设备分支；
   torch 级复用 `kernel/torch_level.py`。
3. **注册选择器**：`register.py` 的 `_BACKENDS` 增加平台名与
   dispatch key 映射（P800：`AutogradCUDA` / `CUDA`）。
4. **设备探测**：`_DEVICE_PROBE` 增加 P800 分支
   （`torch_xmlir` + `torch.cuda.is_available()`）。
5. **三层验证**：`python3 example.py p800-kunlunxin` 一键。
6. **黄金与性能**：跑 `script/check_accuracy.py` 与
   `scripts/perf_run.py --device p800-kunlunxin --update-baseline`。
7. **文档同步**：`PLATFORM.md` / `MERGE.md`、`ops/catalog.yaml`、
   `ops/README.md`、`REPORT.md` 平台列。

## 验证矩阵（result_type / p800-kunlunxin）

| 层 | 命令 | 结果 |
|---|---|---|
| kernel | `python3 test/kernel_level.py p800-kunlunxin` | PASS |
| op | `python3 test/op_level.py p800-kunlunxin` | PASS |
| 应用 | `python3 example.py p800-kunlunxin` | 三层全绿 |
| 黄金 | `python3 script/check_accuracy.py --impl torch --device cuda:1` | 7/7（1 skip），0 fail |
| 性能 | `python3 scripts/perf_run.py --device p800-kunlunxin --pattern "ops.result_type." --update-baseline` | `perf_compare` OK |

## 平台差异与兜底

- P800 无 fp64，float64 黄金用例按设备能力 skip（7 pass / 0 fail / 1 skip）。
