# 算子总体报告: empty_strided

| 项 | 值 |
|---|---|
| 算子名称 | `empty_strided`（分配/元数据类） |
| 语义 | [reference.py](reference.py)：按指定 stride 分配未初始化张量（允许非连续/重叠 stride） |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ 所需存储 empty + as_strided |
| Triton 级 | — | ⬜ 不适用（无数值/设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 16 项元数据检查 |
| 框架层 | `test/op_level.py` | ✅ aten::empty_strided@PrivateUse1（拦截 2 次） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 30/30（元数据） |
| 性能 | 采样 | ✅ 0.0135ms @8192² fp32 |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `empty_strided` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops.empty_strided import register_a1`。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 结论

适用层验证通过（C1 等价 torch API）；因是分配算子，黄金只判定元数据（无数值），哨兵 n/a。

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
