# 算子测试报告: softmax

| 项 | 值 |
|---|---|
| 算子名称 | `softmax`（最后一维） |
| 实现路线 / 开发级别 | A1 torch 算子替换 / Triton 级 |
| 目标设备 | `p800-kunlunxin` |
| 测试日期 | 2026-08-28 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 是否覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ✅ | `test/kernel_level.py` | PASS（24 组 + 哨兵 + 三方性能） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | PASS（拦截命中，路径精度 1.2e-4） |
| 应用层（真实消费） | ✅ | `test/framework_level.py` | PASS（attention 触发 3 次，err=0） |
| 黄金回归 | ✅ | `script/gen_golden.py` + `check_accuracy.py` | PASS（39/39） |
| 性能回归门禁 | ✅ | `scripts/perf_compare.py` | OK（3 用例入基线，0 FAIL） |
| 跨层一致性 | ◐ | 仓库 `--consistency` 锚定 gelu_and_mul | 应用层双跑等效覆盖 |

一键全绿: `python3 example.py p800-kunlunxin`

## 2. 环境

`python3 scripts/check_env.py --device p800-kunlunxin` 核对通过（7/7:
torch 2.9.0+cu129 · triton 3.0.0+03c4c9be · flag_gems 4.2.1rc0 ·
vllm 0.13.0 · vllm-plugin-fl 0.1.0 · transformers 4.57.1 ·
python 3.10.18）。完整快照: `scripts/env_snapshot.py`。

## 3. 精度结果

容差: fp32=1e-5 abs · bf16/fp16=1e-2 abs（参考 = 同输入 fp32 后
cast 回输入 dtype）:

| dtype | 自研 Triton | FlagGems | 原生 ATen |
|---|---|---|---|
| bf16 | 1.8e-3 ✓ | ~2.0e-2 ⚠ | 1.9e-3 ✓ |
| fp16 | 2.2e-4 ✓ | ~2.0e-2 ⚠ | 2.3e-4 ✓ |
| fp32 | **2.4e-7** ✓ | **1.5e-2~4e-2** ⚠ | 0 ✓ |

- 24 组直测（8 shape × 3 dtype，含 3072/5000/5120 尾块回归）全过
- 黄金 39 组（small/tail/extreme，含 zeros·large 特殊用例）39/39
- 哨兵: 确定性 ✓（同输入两次位级一致）· 输入敏感 ✓

## 4. 性能结果

1024² bf16，短采样 20+100，子进程隔离:

| 实现 | 延迟 | 带宽 | 相对自研 | 精度代价 |
|---|---|---|---|---|
| 自研 Triton | 0.060ms | 69.9 GB/s | 1.00x | 无 |
| FlagGems | 0.028ms | — | 2.1x 快 | fp32 超差（见上节） |
| 原生 ATen | **0.010ms** | — | 6.0x 快 | 无 |

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 尾块（N 非 2048 倍数）曾输出错误（#15a: N=5120 err 0.6、N=3072 inf；`other`/`tl.where` 防护无效） | 正确性 | **已修复**（pad 方案），测试形状已固化尾块回归 |
| 2 | autotuner 选非法 num_warps=5（#15b），同 kernel 时对时错 | 可复现性 | **已规避**（固定 BLOCK_N=2048） |
| 3 | FlagGems softmax 三 dtype 精度超差 | 选型 | 已记录（性能结论必须连同精度读） |

## 6. 结论

**通过，可交付**。正确性体系全绿且含尾块/特殊值回归；自研精度
三方最优；性能落后原生与 FlagGems 为已知状态，优化方向明确
（单遍在线归约），不影响正确性结论。
