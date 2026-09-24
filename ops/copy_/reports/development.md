# 算子开发报告: copy_

| 项 | 值 |
|---|---|
| 算子名称 | `copy_` |
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

`aten::copy_(Tensor self, Tensor src, bool non_blocking=False) -> Tensor`：
**原地**把 src 写入 self 并返回 self。src 可广播到 self.shape，元素按
self.dtype cast。纯搬运无算术，原生即位级标准。

### 2.2 参考与接口

[reference.py](../reference.py)（原生 `dst.copy_(src)`）；
Triton 级 `copy_triton(dst, src, non_blocking)`。

### 2.3 数值规格

dst/src bf16/fp16/fp32；cast 位级一致；容差按 **dst dtype** 分档。
`boundary` 特殊用例会合法溢出到 ±inf（fp32→fp16），判定用 isclose
（inf 相等、nan 相等）。

## 3. 实现说明

[kernel/triton_level.py](../kernel/triton_level.py)：连续侧扁平 kernel；
广播经 `src.expand`；非连续 `_collapse` 后 rank 1–4；**2D 转置型走
`_copy_2d_swiz` 分块**（端口自 copy_r3）；自拷贝/别名/溢出/不支持 dtype
走原生回退。**#11** device 上下文；无归约不涉 #15a；不用 autotune。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py) |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py) |
| 硬件级 | ⬜ 置空 | [kernel/hardware_level/README.md](../kernel/hardware_level/README.md) |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 4 shape × 9 dtype 对 + 广播 + 非连续 + 自拷贝；最差 err 0；哨兵 ✓ |
| 框架层 | ✅ | `aten::copy_@PrivateUse1`，拦截 4 次，cast/广播/回退正确 |
| 应用层 | ✅ | 子进程命中 1 次，输出逐位一致 |
| 黄金 | ✅ | 78/78（same/cast/broadcast 三类） |

## 5. 性能

8192² fp32 同形原地：

| 实现 | 延迟 | 带宽 |
|---|---|---|
| 自研 Triton | 0.242 ms | 2215 GB/s |
| torch 级 | 0.237 ms | 2261 GB/s |
| reference（原生） | **0.237 ms** | 2261 GB/s |

与原生持平（差 ~2%）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | copy_ 为极高频内部算子，全局注册有侵入性 | 需子进程隔离编排 | 已在 register.py 注明 |
| 2 | rank≥3 非连续未分块 | 较慢 | 待优化 |
| 3 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论与后续

三层全绿，78/78 黄金，原地/广播/cast 语义正确，性能与原生持平。
后续: 高 rank 分块、真实 vLLM 应用层。

## 附录: 复现命令

```bash
cd ops/copy_
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```
