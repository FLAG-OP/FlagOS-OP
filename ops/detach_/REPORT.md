# 算子总体报告: detach_

| 项 | 值 |
|---|---|
| 算子名称 | `detach_`（别名/元数据（autograd）类） |
| 语义 | [reference.py](reference.py)：原地脱离 autograd，返回 self (requires_grad=False) |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；register 契约保留，但该算子不可 A1 注册） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **自用/实验** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | — | ⬜ 不适用（无设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键验证 | `python3 example.py cambricon` | ✅（适用层） |
| kernel 层 | `test/kernel_level.py` | ✅ 24 项 |
| 框架层 | `test/op_level.py` | N/A（detach_ 非 backend dispatch，不可 A1 注册） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 36/36 |
| 性能 | 采样 | ✅ 0.0012ms |

## 2026-10-12 注册契约同步

- `register_a1` 已对齐 `counter` / `platform` 签名。
- 该算子保持 `NotImplementedError`，不强行做 A1 注册。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 结论

适用层验证通过；原地返回 self + requires_grad=False + 存储不变。无 Triton/硬件级（该类算子固有），注册/框架层为 N/A。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)

## 证据分级（C0 / C1 / C2，对齐 #42 P800）

| 级别 | 含义 | 本 PR |
|---|---|---|
| **C0** | 黄金 + op 层 | ✅ |
| **C1** | per-op 等价消费方（等价 torch API 子进程） | ✅ |
| **C2** | 真实 vLLM 前向实际命中 | ✖ 受 known-issues #26 限制，不做声明 |

结论：本 PR 达到 **C1**，C2 不作交付声明。
