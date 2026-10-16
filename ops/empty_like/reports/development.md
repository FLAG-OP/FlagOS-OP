# 算子开发报告: empty_like

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子名称 | `empty_like`（分配/元数据类） |
| 实现路线 / 开发级别 | **A1** / **torch 级**（无 Triton/硬件级） |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

> **2026-10-12 维护记录**：`register_a1` 已升级为 `counter + platform` 契约；canonical key 为 `AutogradPrivateUse1`，历史 `PrivateUse1` 结论保留为兼容 key。实现导入已改为包态优先、单算子脚本态 fallback。见 PR #25。

## 1. 环境配置

MLU590-M9 ×8；torch 2.7.1+cpu / torch_mlu 1.29.2 / triton 3.2.0+mlu /
flag_gems 5.3.5；dispatch key `PrivateUse1`；vLLM 未安装。
`check_env --device cambricon` → OK。

## 2. 算子定义

`aten::empty_like(Tensor self, *, dtype, layout, device, pin_memory, memory_format) -> Tensor`：
分配未初始化内存，shape 同 self；dtype/layout/device 默认继承 self，
memory_format 默认 preserve_format。**正确性只由元数据定义**。

## 3. 实现说明

[kernel/torch_level.py](../kernel/torch_level.py)：`_target_strides` 用
meta 张量推导目标 layout 的 stride（preserve_format 取 `self.to("meta")`，
显式格式用 meta empty），再 `torch.empty_strided` 分配。
[kernel/README.md](../kernel/README.md) 说明 Triton/硬件级为何不适用。

| 级别 | 状态 |
|---|---|
| torch 级 | ✅ |
| Triton 级 | ⬜ 不适用 |
| 硬件级 | ⬜ 不适用 |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 4 shape × 3 dtype × 2 view × 2 memory_format + channels_last + 零元素 = 50 项 |
| 框架层 | ✅ | `aten::empty_like@PrivateUse1`，拦截 3 次，元数据一致 |
| 应用层 | ✅ | 子进程命中 1 次，输出逐位一致 |
| 黄金 | ✅ | 168/168（元数据：shape/dtype/stride） |

## 5. 性能

8192² fp32 分配：`torch` 0.0133ms（分配器受限，仅可用性采样）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 数值未初始化 | 不能做数值断言 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论

适用层验证通过（C1 等价 torch API），元数据 168/168 一致；无 Triton/硬件级（分配算子固有）。
后续: 真实 vLLM 应用层（与其他算子统一做）。

## 附录: 复现命令

```bash
cd ops/empty_like
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
```
