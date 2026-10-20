# 平台绑定清单（Platform Binding）— detach_

> 当前交付包含两个平台：**ascend910**（Ascend 910 · torch_npu 2.10 ·
> flagtree/triton 3.5.1 · CANN 9.0）与 **cambricon**（MLU590 · torch_mlu）。
> 本文件记录不可凭直觉搬运的绑定结论；跨平台移植时逐项重验，不要把单平台
> 实验结果当成普适行为。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| 语义参考 | `reference.py`（判卷标准） |
| 黄金规格与数据 | `goldendata/`（CPU 生成，跨平台） |
| 三层测试、性能脚本 | `test/`、`script/` |
| A1 注册（多后端选择器 + 成对 key） | `register.py` |

## 2. ascend910 绑定

| 绑定项 | 实测结论 |
|---|---|
| A1 拦截 key | **AutogradPrivateUse1**（torch_npu 的 C++ Autograd 包装不 redispatch，仅注册 PrivateUse1 永不命中——见 `../sdpa/PLATFORM.md` §2） |
| 注册键成对覆盖 | 同进程同时注册 AutogradPrivateUse1 + PrivateUse1（`sdpa_math` 范式）；工厂算子实际落 PrivateUse1 |
| 设备同步 | `torch.npu.synchronize`（非 cuda） |
| task queue 排序 | 镜像默认 `TASK_QUEUE_ENABLE=1`：torch 算子入 task queue 异步执行，triton-ascend 直发 raw stream，两者**无顺序保证** → 自研 kernel 读到未完成输入（非确定性错值/NaN）。`common/device.py` 在 torch_npu 初始化前强制 `TASK_QUEUE_ENABLE=0`。见 `docs/known-issues.md` #24 |
| 路线 | 按设计**不可** A1 注册（原地 detach 破坏 autograd 语义），保持 `NotImplementedError`；仅 kernel 层 + 应用式前向验证 |
|
| 语义参考 | 见下（平台无关） |

## 3. cambricon（MLU590）绑定

- profile `cambricon`：`torch_device=mlu:0`、`dispatch_key=PrivateUse1`；
- `_BACKENDS` 按可用设备消歧（同机双 NPU+MLU 时须显式 `platform=`）；
- A1 注册键成对覆盖（同 §2）。

## 4. 新平台接入

按 `MERGE.md` 七步走；kernel 层实现 + `PLATFORM` + `_DEVICE_PROBE` 四件并入
`_BACKENDS`。

---
Dashener2 2026-10-15-fix（Ascend 移植）
# Dashener2 <dashen272@outlook.com>
