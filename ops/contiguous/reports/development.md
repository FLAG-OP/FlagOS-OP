# 算子开发报告: contiguous

| 项 | 值 |
|---|---|
| 算子名称 | `contiguous` |
| 实现路线 / 开发级别 | **A1** / **Triton 级**（+ torch 级对照） |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 1. 环境配置

| 项 | 值 |
|---|---|
| 硬件 | MLU590-M9 ×8（98304 MiB/卡），firmware v1.5.0 / CNMON v6.2.29 |
| PyTorch / torch_mlu | 2.7.1+cpu / 1.29.2+torch2.7.1 |
| Triton / FlagGems | 3.2.0+mlu1.7.2 / 5.3.5 |
| dispatch key | `PrivateUse1`；vLLM/transformers 未安装 |

`check_env --device cambricon` → OK。

## 2. 算子定义

### 2.1 语义

`aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor`：
返回在 memory_format 下连续的张量；**已连续则返回 self 本身（别名）**，
否则分配新张量搬运。本次 Triton 路径覆盖 `contiguous_format`，其余
memory_format（channels_last 等）回退原生。

### 2.2 参考与接口

[reference.py](../reference.py)（原生 `.contiguous`）；Triton 级
`contiguous_triton(inp, memory_format)`。

### 2.3 数值规格

bf16/fp16/fp32；纯搬运无算术，位级一致；容差 fp32 1e-5 / fp16,bf16 1e-2。

## 3. 实现说明

[kernel/triton_level.py](../kernel/triton_level.py)：

- 别名快路径 `is_contiguous(memory_format)` → 返回 self。
- 非连续侧 `_collapse` 后 rank 1–4 strided kernel。
- **转置型 2D 加速**: 端口自 `copy_r3` 的 `_contig_2d_swiz` 分块 kernel
  （src/dst 最内维都连续 → burst 访问）。加之前 8192² 转置拷贝
  24.7ms（21 GB/s，逐元素 gather），加之后 **0.30ms（1762 GB/s）**，
  提升约 80x。
- **#11** device 上下文；无归约不涉 #15a；不用 autotune。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py) |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py) |
| 硬件级 | ⬜ 置空 | [kernel/hardware_level/README.md](../kernel/hardware_level/README.md) |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 别名快路径 + 4 视图 + 2D 非整倍数；最差 err 0；哨兵 ✓ |
| 框架层 | ✅ | `aten::contiguous@PrivateUse1`，拦截 5 次，别名/回退正确 |
| 应用层 | ✅ | 子进程命中 1 次，输出逐位一致 |
| 黄金 | ✅ | 114/114（contig + view 两种 layout） |

## 5. 性能

8192² fp32 转置→连续：

| 实现 | 延迟 | 带宽 |
|---|---|---|
| 自研 Triton（swiz 后） | 0.305 ms | 1762 GB/s |
| torch 级 | 0.254 ms | 2112 GB/s |
| reference（原生） | **0.254 ms** | 2116 GB/s |

自研较原生慢约 20%（分块 swiz vs 厂商优化拷贝）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | rank≥3 非连续仍走逐元素 gather | 慢 | 待扩展 swiz 到更高 rank |
| 2 | channels_last 等走原生回退 | 无自研覆盖 | 已知 |
| 3 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论与后续

三层全绿，114/114 黄金，别名语义正确，转置性能经 swiz 修复后接近原生。
后续: 高 rank 分块、channels_last 自研、真实 vLLM 应用层。

## 附录: 复现命令

```bash
cd ops/contiguous
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```
