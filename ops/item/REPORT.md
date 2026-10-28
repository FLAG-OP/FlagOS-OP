# 算子总体报告: item

| 项 | 值 |
|---|---|
| 算子名称 | `item`（host 标量类） |
| 语义 | [reference.py](reference.py)：单元素张量取为 Python 标量 (device→host 同步) |
| 硬件平台 | **2 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`）· p800-kunlunxin（P800 / XMLIR；A1 canonical `AutogradCUDA`，兼容 `CUDA`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ `p800-kunlunxin`（P800）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 / 2026-10-22 P800 交付 / 2026-10-26 复验 |

## p800-kunlunxin 接入（2026-10-22 交付 / 2026-10-26 复验）

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | torch |
| 三层验证 | `python3 example.py p800-kunlunxin` 全绿；op 层拦截 1 次 |
| 黄金 | 36/36，0 fail |
| 性能门禁 | 空闲 P800 两轮复测：`perf_compare --device p800-kunlunxin` = OK（FAIL 0 / WARN 0 / NEW 0） |

| 性能用例 | 延迟 (ms) |
|---|---:|
| reference | 0.014 |
| torch | 0.016 |

**P800 绑定结论**：reference 0.0142 ms / torch 0.0160 ms；device→host 标量同步开销为主。

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
| kernel 层 | `test/kernel_level.py` | ✅ 12 项 |
| 框架层 | `test/op_level.py` | ✅ aten::item@PrivateUse1（拦截 1 次） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 36/36 |
| 性能 | 采样 | ✅ 0.0433ms |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `item` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops.item import register_a1`。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 结论

适用层验证通过（C1 等价 torch API）；标量值相等 + Python 类型 + 多元素报错。无 Triton/硬件级（该类算子固有）。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
