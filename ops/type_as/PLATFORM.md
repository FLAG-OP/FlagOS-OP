# 平台绑定清单（Platform Binding）— type_as

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
| 实现路径 | A1 · Triton |
| 黄金 | 111/111，0 fail |
| 性能基线 | `perf/baselines/cambricon.json` |

## 3. p800-kunlunxin 绑定

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | Triton |
| 黄金 | 111/111，0 fail |
| 性能基线 | `perf/baselines/p800-kunlunxin.json` |

**P800 绑定结论**：目标 bf16 在 P800 走原生 type_as 兜底：fp16→bf16 store 实测 misaligned，fp32→bf16 有舍入偏差；其余 dtype 走 Triton。

## 4. ascend910 绑定（证据级别 C1）

| 项 | 结论 |
|---|---|
| A1 拦截 key | `AutogradPrivateUse1`（成对覆盖 `PrivateUse1`；工厂算子实际落 `PrivateUse1`） |
| 设备探测 | `import torch_npu` + `torch.npu.is_available()` |
| 实现路径 | A1 · Triton |
| Triton tile 上限 | Ascend UB 192KB：连续 16384 / strided 8192（超出触发 BiShengHIR `ub overflow`）；`_caps(device.type)` 自适应 |
| channels_last | torch_npu 不支持；测试在 npu 跳过该 view |
| task queue 排序 | 镜像默认 `TASK_QUEUE_ENABLE=1`，torch 异步队列与 triton-ascend 直发 raw stream 无序；`common/device.py` 在 Ascend 显式 opt-in 下置 0（[#24](../../docs/known-issues.md#24)） |
| 证据级别 | **C1**：注册正常但参考 tiny 模型前向未调用（count=0），未能证明命中 |

## 5. 证据分级（C0 / C1 / C2，对齐 #42 P800）

| 级别 | 含义 | 本 PR |
|---|---|---|
| **C0** | 黄金 + op 层 | ✅ |
| **C1** | per-op 等价消费方（等价 torch API 子进程） | ✅ |
| **C2** | 真实 vLLM 前向实际命中 | ✖（注册正常但参考 tiny 模型前向未调用（count=0），未能证明命中） |

本算子达到 **C1**（注册正常但参考 tiny 模型前向未调用（count=0），未能证明命中），C2 不作声明。

### 复现与替代路径

复现: `python3 scripts/c2_enginecore/validate.py --op type_as --model /tmp/c2model`（生成阶段 count=0 → 验收非零退出）。替代路径: 换会调用 `type_as` 的模型/输入后重判。

## 6. 新平台接入

按 `MERGE.md` 七步走；`register.py` 的 `_BACKENDS` 加一行，补该平台的
`_DEVICE_PROBE` 与 kernel/实现选择即可。
