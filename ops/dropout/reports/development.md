# 算子开发报告: dropout

| 项 | 值 |
|---|---|
| 算子名称 | `dropout` |
| 实现路线 / 开发级别 | **A1** / **Triton 级**（Philox RNG） |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 1. 环境配置

| 项 | 值 |
|---|---|
| 硬件 | MLU590-M9 ×8，firmware v1.5.0 / CNMON v6.2.29 |
| PyTorch / torch_mlu | 2.7.1+cpu / 1.29.2+torch2.7.1 |
| Triton / FlagGems | 3.2.0+mlu1.7.2 / 5.3.5 |
| dispatch key | `PrivateUse1`；vLLM/transformers 未安装 |

## 2. 算子定义

### 2.1 语义

`aten::dropout(Tensor input, float p, bool train) -> Tensor`：
- p ∉ [0,1] → RuntimeError；
- train=False 或 p=0 → 返回 input 本身（别名）；
- p=1 → 全零（suggested format）；
- 0<p<1 且 train=True → `out = input * mask/(1-p)`，mask ~ Bernoulli(1-p)。

### 2.2 参考与判定口径

[reference.py](../reference.py)（原生 `F.dropout`）。**关键**: Triton 的
Philox 流与 PyTorch 原生 dropout 不同，随机分支**不能逐元素比较**：
- 确定性分支（train=False / p=0 / p=1）→ 位级判定；
- 随机分支 → 结构 + 统计判定（值域 {0, x/(1-p)}、drop 比例、seed 可控）。

### 2.3 数值规格

bf16/fp16/fp32；容差 fp32 1e-5 / fp16,bf16 1e-2；drop 比例容差 0.02。

## 3. 实现说明

[kernel/triton_level.py](../kernel/triton_level.py)：`tl.philox` 计数器 RNG
（镜像 FlagGems），读并推进 `torch.mlu.default_generators` 状态使
`torch.manual_seed` 可控；`_suggest_memory_format` 复刻 ATen layout 决策；
channels_last 直用、非连续回退。**#11** device 上下文；无归约不涉 #15a；
不用 autotune。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py) |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py) |
| 硬件级 | ⬜ 置空 | [kernel/hardware_level/README.md](../kernel/hardware_level/README.md) |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 确定性 3 分支 + 非法 p；随机 4 shape × 3 dtype × 3 p 结构/统计；最大 drop 偏差 0.0016 |
| 框架层 | ✅ | `aten::dropout@PrivateUse1`，拦截 5 次，随机分支结构验证 |
| 应用层 | ✅ | 子进程命中 1 次（eval 透传），输出逐位一致 |
| 黄金 | ✅ | 189/189（确定性位级 + 随机结构统计） |

## 5. 性能

8192² fp16 p=0.5：

| 实现 | 延迟 | 带宽 |
|---|---|---|
| 自研 Triton | 2.785 ms | 96 GB/s |
| FlagGems `dropout` | 2.960 ms | 90 GB/s |
| torch 级 | 0.563 ms | 476 GB/s |
| reference（原生） | **0.563 ms** | 476 GB/s |

**RNG 受限**：Philox 生成 + 4-unroll 访存的 Triton 实现比原生慢约 5x，
但与 FlagGems 同机制同级（略快 ~6%）。原生使用硬件优化 RNG。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | Triton Philox 比原生 RNG 慢 ~5x | 性能 | 机制限制，FlagGems 同样 |
| 2 | RNG 流与原生不同 | 无法位级对齐随机分支 | 语义允许（dropout 本身随机） |
| 3 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论与后续

三层全绿，189/189 黄金（确定性位级 + 随机统计），RNG 可由
`torch.manual_seed` 控制。性能受 Philox 限制，比原生慢 ~5x，与
FlagGems 持平。后续: 真实 vLLM 应用层、RNG 向量化/更高 unroll 调优。

## 附录: 复现命令

```bash
cd ops/dropout
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```
