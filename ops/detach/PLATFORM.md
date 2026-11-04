# 平台绑定清单（Platform Binding）— detach

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
| 实现路径 | A1 · torch |
| 黄金 | 36/36，0 fail |
| 性能基线 | `perf/baselines/cambricon.json` |

## 3. p800-kunlunxin 绑定

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | torch |
| 黄金 | 36/36，0 fail |
| 性能基线 | `perf/baselines/p800-kunlunxin.json` |

**P800 绑定结论**：共享存储的非原地视图，P800 直接复用。

## 4. ascend910 绑定（证据级别 C1）

| 项 | 结论 |
|---|---|
| A1 拦截 key | `AutogradPrivateUse1`（成对覆盖 `PrivateUse1`；工厂算子实际落 `PrivateUse1`） |
| 设备探测 | `import torch_npu` + `torch.npu.is_available()` |
| 实现路径 | torch 级 |
| autograd | `aten::detach` 落 `AutogradPrivateUse1`；实现保持共享存储视图语义 |
| task queue 排序 | 镜像默认 `TASK_QUEUE_ENABLE=1`，torch 异步队列与 triton-ascend 直发 raw stream 无序；`common/device.py` 在 Ascend 显式 opt-in 下置 0（[#24](../../docs/known-issues.md#24)） |
| 证据级别 | **C1**：注册破坏 nn.Parameter/FakeTensor 构造（已知 vLLM 不兼容） |

## 5. 证据分级（C0 / C1 / C2，对齐 #42 P800）

| 级别 | 含义 | 本 PR |
|---|---|---|
| **C0** | 黄金 + op 层 | ✅ |
| **C1** | per-op 等价消费方（等价 torch API 子进程） | ✅ |
| **C2** | 真实 vLLM 前向实际命中 | ✖（注册破坏 nn.Parameter/FakeTensor 构造（已知 vLLM 不兼容）） |

本算子达到 **C1**（注册破坏 nn.Parameter/FakeTensor 构造（已知 vLLM 不兼容）），C2 不作声明。

### 复现与替代路径

复现: `python3 scripts/c2_enginecore/validate.py --op detach --model /tmp/c2model`（`FakeTensor ... already associated ...`）。替代路径: 不在 vLLM 进程内注册 detach（保持 kernel/op 层验证）。

## 6. 新平台接入

按 `MERGE.md` 七步走；`register.py` 的 `_BACKENDS` 加一行，补该平台的
`_DEVICE_PROBE` 与 kernel/实现选择即可。
