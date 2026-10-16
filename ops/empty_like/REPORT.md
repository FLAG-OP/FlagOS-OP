# 算子总体报告: empty_like

| 项 | 值 |
|---|---|
| 算子名称 | `empty_like`（分配/元数据类） |
| 语义 | [reference.py](reference.py)：未初始化，只约束 shape/dtype/stride/device |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 |

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
| kernel 层 | `test/kernel_level.py` | ✅ 50 项元数据检查 |
| 框架层 | `test/op_level.py` | ✅ 拦截 3 次 |
| 应用层 | `test/framework_level.py` | ✅（等价 torch API） |
| 黄金 | `script/` | ✅ 168/168（元数据） |
| 性能 | 采样 | ✅ `torch` 0.0133ms @8192² fp32 |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `empty_like` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops.empty_like import register_a1`。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 结论

适用层验证通过（C1 等价 torch API）；因是分配算子，黄金只判定元数据（无数值），哨兵 n/a。
与 FlagGems 差异：不引入「写 0 dummy kernel」以满足 device code 要求。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
