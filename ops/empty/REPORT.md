# 算子总体报告: empty

| 项 | 值 |
|---|---|
| 算子名称 | `empty`（分配/元数据类） |
| 语义 | [reference.py](reference.py)：分配 size 形状的未初始化张量，只保证元数据 |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ meta 推 stride + empty_strided |
| Triton 级 | — | ⬜ 不适用（无数值/设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 15 项元数据检查 |
| 框架层 | `test/op_level.py` | ✅ — |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 48/48（元数据） |
| 性能 | 采样 | ✅ 0.0107ms @8192² fp32 |

## 结论

三层全绿；因是分配算子，黄金只判定元数据（无数值），哨兵 n/a。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
