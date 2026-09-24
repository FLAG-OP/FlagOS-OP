# 算子总体报告: result_type

| 项 | 值 |
|---|---|
| 算子名称 | `result_type`（dtype 元数据类） |
| 语义 | [reference.py](reference.py)：两输入在 PyTorch 类型提升下的公共 dtype |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **自用/实验** |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | — | ⬜ 不适用（无设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 64 项 |
| 框架层 | `test/op_level.py` | ✅ —（result_type 为纯函数，不可 A1 注册） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 8/8 |
| 性能 | 采样 | ✅ 0.0013ms |

## 结论

三层全绿；公共 dtype 相等。无 Triton/硬件级（该类算子固有）。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
