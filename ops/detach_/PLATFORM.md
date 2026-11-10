# 平台绑定清单（Platform Binding）— detach_

> 当前交付包含三个平台：**cambricon**（MLU590）、**p800-kunlunxin**（昆仑芯 P800 / XMLIR）与 **ascend910**（Ascend 910）。
> 本文件记录不可凭直觉搬运的绑定结论；跨平台移植时逐项重验。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| 语义参考 | `reference.py` |
| 黄金规格与数据 | `goldendata/`（CPU 生成，天然跨平台） |
| 三层测试与性能脚本 | `test/`、`script/` |
| A1 注册选择器 | `register.py`（`_BACKENDS` + `_DEVICE_PROBE`） |

## 2. cambricon（MLU590）绑定

| 项 | 结论 |
|---|---|
| A1 拦截 key | `AutogradPrivateUse1`（兼容 `PrivateUse1`） |
| 设备探测 | `import torch_mlu` + `torch.mlu.is_available()` |
| 实现路径 | experimental · torch |
| 黄金 | 36/36，0 fail |
| 性能基线 | `perf/baselines/cambricon.json` |

## 3. p800-kunlunxin 绑定

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | torch（不注册） |
| 黄金 | 36/36，0 fail |
| 性能基线 | `perf/baselines/p800-kunlunxin.json` |

**P800 绑定结论**：按设计不可 A1 注册（autograd key 语义），P800 仅 kernel/应用等价验证。

## 4. ascend910 绑定（证据级别 C0）

| 项 | 结论 |
|---|---|
| A1 拦截 key | 不适用；按设计不注册，`register_a1` 保持 `NotImplementedError` |
| 设备探测 | `import torch_npu` + `torch.npu.is_available()` |
| 实现路径 | torch 级 |
| 路线 | 按设计不可 A1 注册（原地 detach），保持 `NotImplementedError`；仅 kernel/op 层 + 原生应用 smoke |
| task queue 排序 | 镜像默认 `TASK_QUEUE_ENABLE=1`，torch 异步队列与 triton-ascend 直发 raw stream 无序；`common/device.py` 在 Ascend 显式 opt-in 下置 0（[Ascend task queue](../../docs/known-issues.md#ascend-task-queue)） |
| 证据级别 | **C0**（原生应用 smoke；未消费仓库实现） |

## 5. ascend910 证据分级（C0 / C1 / C2）

| 级别 | 含义 | 本 PR |
|---|---|---|
| **C0** | 黄金 + op 层 | ✅ |
| **C1** | 消费仓库实现的等价消费方 | ✖（当前仅原生 torch smoke） |
| **C2** | 真实 vLLM 前向实际命中 | ✖（不适用：未消费仓库实现） |

本算子按 **C0**（原生应用 smoke）登记：framework 测试只调原生 torch API，未消费仓库实现；不声明 C1/C2。

### 复现与替代路径

不可 A1 注册（原地 detach，by design）。替代路径: 仅 kernel 层 + 等价应用式前向 smoke；如需 A1 需 vLLM 侧支持。

## 6. 新平台接入

按 `MERGE.md` 七步走；`register.py` 的 `_BACKENDS` 加一行，补该平台的
`_DEVICE_PROBE` 与 kernel/实现选择即可。
