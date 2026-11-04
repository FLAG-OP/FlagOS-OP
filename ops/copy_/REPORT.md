# 算子总体报告: copy_

| 项 | 值 |
|---|---|
| 算子名称 | `copy_` |
| 语义 | [reference.py](reference.py)（原生 `dst.copy_(src)`，原地，位级标准） |
| 硬件平台 | **2 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`）· p800-kunlunxin（P800 / XMLIR；A1 canonical `AutogradCUDA`，兼容 `CUDA`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ `p800-kunlunxin`（P800）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 / 2026-10-22 P800 交付 / 2026-10-26 复验 |

## p800-kunlunxin 接入（2026-10-22 交付 / 2026-10-26 复验）

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | Triton（P800 dst=bf16 且 src 非 bf16 时先原生 type_as 转换） |
| 三层验证 | `python3 example.py p800-kunlunxin` 全绿；op 层拦截 4 次 |
| 黄金 | 78/78，0 fail |
| 性能门禁 | 空闲 P800 两轮复测：`perf_compare --device p800-kunlunxin` = OK（FAIL 0 / WARN 0 / NEW 0） |

| 性能用例 | 延迟 (ms) |
|---|---:|
| reference | 0.245 |
| torch | 0.365 |
| triton | 0.482 |

**P800 绑定结论**：自研 Triton 0.482 ms / 原生 0.245 ms（约 0.51x）；bf16 写出兜底路径功能正确，性能瓶颈同 clone。

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ | — |
| kernel 层 | `test/kernel_level.py` | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ✅ | — |
| 应用层 | `test/framework_level.py` | ✅（等价 torch API） | — |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 78/78 | [goldendata/](goldendata/README.md) |
| 性能 | perf 门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `copy_` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops.copy_ import register_a1`。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 关键数字

| 指标 | 自研 Triton | 原生 |
|---|---|---|
| 精度（78 组） | **0**（位级一致） | 0 |
| 8192² fp32 原地拷贝 | 0.242 ms | **0.237 ms** |
| 原地返回 / 广播 / 哨兵 | ✅ / ✅ / ✅ | — |

## A1 注入排除边界（2026-10-29，known-issues #26）

P800 bf16 兜底路径会经 `aten::type_as` / `Tensor.to` 转换，而两者内部
重新进入已注册的 `copy_`；`FLAGOS_A1_OPS` 同时含 `copy_` 与 `type_as`
时形成 `copy_ ↔ type_as/to` 无限递归（4.2 真实 vLLM C2 实测）。

因此 `copy_` **不进入真实 vLLM 编排的 A1 注入清单**：

- 框架级/真实引擎 C2 跑法的 `FLAGOS_A1_OPS` 默认排除 `copy_`；
- 单算子三层/黄金/性能验证不受影响（这些路径不经 vLLM 编排）；
- 若未来要放开，需先在 `type_as`/`to` 侧切断对 `copy_` 的再入
  （或改注册为「已注册不拦截」），不能仅靠约定。

## 结论与遗留

适用层验证通过（C1 等价 torch API）、黄金 78/78、原地/广播/cast 语义正确，性能与原生持平。
遗留: 高 rank 分块、真实 vLLM 应用层。注意 copy_ 全局注册侵入性强，
编排里框架层用子进程隔离。

## A1 注入排除边界（C2 实测，2026-10-26/#26）

P800 bf16 兜底路径会经 `aten::type_as` / `Tensor.to` 转换，而两者内部
重新进入已注册的 `copy_`；`FLAGOS_A1_OPS` 同时含 `copy_` 与 `type_as`
时形成 `copy_ ↔ type_as/to` 无限递归（4.2 栈真实 vLLM C2 实测）。

边界约定：

- 框架级 `FLAGOS_A1_OPS` 默认列表**不含** `copy_`（与 `empty`/`detach_`/
  `result_type` 同类「按设计不注入」）；
- 需要等价 cast 时改用不经过 `copy_` 的路径（如 `type_as` 单独注入、
  或 `torch.empty` + 原生拷贝）；
- 5.0 栈本轮 C2（#39）未注入 `copy_`，未触发；
- 调试时若怀疑递归，用 `A1_OPS_DEBUG=1` 看 A1 dispatch 日志里
  `copy_` 是否被 `type_as/to` 间接重入。

详见 [known-issues #26](../../docs/known-issues.md)。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
