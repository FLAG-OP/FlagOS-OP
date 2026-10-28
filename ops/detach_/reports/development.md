# 算子开发报告: detach_

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子名称 | `detach_`（别名/元数据（autograd）类） |
| 实现路线 / 开发级别 | **自用/实验** / **torch 级** |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

> **2026-10-12 维护记录**：`register_a1` 已升级为 `counter + platform` 契约；canonical key 为 `AutogradPrivateUse1`，历史 `PrivateUse1` 结论保留为兼容 key。该算子仍保持 `NotImplementedError`，不强行注册。见 PR #25。

## 1. 环境配置

MLU590-M9 ×8；torch 2.7.1+cpu / torch_mlu 1.29.2 / triton 3.2.0+mlu /
flag_gems 5.3.5；dispatch key `PrivateUse1`；vLLM 未安装。
`check_env --device cambricon` → OK。

## 2. 算子定义

`aten::detach_`：原地脱离 autograd，返回 self (requires_grad=False)。判定口径：原地返回 self + requires_grad=False + 存储不变。

## 3. 实现说明

[kernel/torch_level.py](../kernel/torch_level.py)；Triton/硬件级不适用
（见 [kernel/README.md](../kernel/README.md)）。

| 级别 | 状态 |
|---|---|
| torch 级 | ✅ |
| Triton 级 | ⬜ 不适用 |
| 硬件级 | ⬜ 不适用 |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 4 shape × 3 dtype × {requires_grad True/False}（24 项） |
| 框架层 | ✅ | —（detach_ 非 backend dispatch，不可 A1 注册） |
| 应用层 | ✅ | 子进程前向正常/一致 |
| 黄金 | ✅ | 36/36 |

## 5. 性能

0.0012ms（该类算子为元数据/同步操作，非带宽/算力受限）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无设备码 | 无 Triton 实现 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论

适用层验证通过，36/36 黄金通过；注册/框架层为 N/A。后续：真实 vLLM 应用层。

## P800（p800-kunlunxin）复验（2026-10-26）

- 平台呈现：XMLIR 将 XPU 呈现为 CUDA（`device.type == "cuda"`）。
- A1 拦截：按设计不注册（自用/实验路线）。
- 实现路径：torch（不注册）。
- 三层验证：kernel 与应用等价路径全绿；按设计不做 A1 注册。
- 黄金：36/36，0 fail。
- 性能：`perf_compare --device p800-kunlunxin` = OK（FAIL 0 / WARN 0 / NEW 0）。
- 绑定细节与平台差异见 [PLATFORM.md](../PLATFORM.md)，合并流程见 [MERGE.md](../MERGE.md)。

## 附录: 复现命令

```bash
cd ops/detach_
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
```
