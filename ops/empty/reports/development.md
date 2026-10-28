# 算子开发报告: empty

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子名称 | `empty`（分配/元数据类） |
| 实现路线 / 开发级别 | **A1** / **torch 级**（无 Triton/硬件级） |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

> **2026-10-12 维护记录**：`register_a1` 已升级为 `counter + platform` 契约；canonical key 为 `AutogradPrivateUse1`，历史 `PrivateUse1` 结论保留为兼容 key。该算子仍保持 `NotImplementedError`，不强行注册。见 PR #25。

## 1. 环境配置

MLU590-M9 ×8；torch 2.7.1+cpu / torch_mlu 1.29.2 / triton 3.2.0+mlu /
flag_gems 5.3.5；dispatch key `PrivateUse1`；vLLM 未安装。
`check_env --device cambricon` → OK。

## 2. 算子定义

`aten::empty`：分配 size 形状的未初始化张量，只保证元数据。**正确性只由元数据定义**。

## 3. 实现说明

[kernel/torch_level.py](../kernel/torch_level.py)：meta 推 stride + empty_strided。
[kernel/README.md](../kernel/README.md) 说明 Triton/硬件级为何不适用。

| 级别 | 状态 |
|---|---|
| torch 级 | ✅ |
| Triton 级 | ⬜ 不适用 |
| 硬件级 | ⬜ 不适用 |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 15 项元数据检查 |
| 框架层 | ✅ | — |
| 应用层 | ✅ | 子进程前向正常/逐位一致 |
| 黄金 | ✅ | 48/48（元数据） |

## 5. 性能

8192² fp32 分配：0.0107ms（分配器受限，仅可用性采样）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 数值未初始化 | 不能做数值断言 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论

适用层验证通过，元数据 48/48 一致；无 Triton/硬件级（分配算子固有），注册/框架层为 N/A。

## P800（p800-kunlunxin）复验（2026-10-26）

- 平台呈现：XMLIR 将 XPU 呈现为 CUDA（`device.type == "cuda"`）。
- A1 拦截：按设计不注册（自用/实验路线）。
- 实现路径：torch（不注册）。
- 三层验证：kernel 与应用等价路径全绿；按设计不做 A1 注册。
- 黄金：48/48，0 fail。
- 性能：`perf_compare --device p800-kunlunxin` = OK（FAIL 0 / WARN 0 / NEW 0）。
- 绑定细节与平台差异见 [PLATFORM.md](../PLATFORM.md)，合并流程见 [MERGE.md](../MERGE.md)。

## 附录: 复现命令

```bash
cd ops/empty
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
```
