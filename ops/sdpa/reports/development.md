# 算子开发报告: sdpa

| 项 | 值 |
|---|---|
| 算子名称 | `aten::scaled_dot_product_attention` |
| 实现路线 | **A1** torch 算子替换 |
| 开发级别 | **Triton 级**（自研设备码） |
| 目标设备 | `ascend910`（Ascend 910_9382, 24 AICore/卡, CANN 9.0.0） |
| 开发者 | wt \<wangt635@ustc.edu.cn\> |
| 日期 | 2026-09-16 |
| 报告状态 | 定稿 |

---

## 1. 环境配置

### 1.1 硬件

| 项 | 值 |
|---|---|
| 设备 | Ascend 910_9382 ×8（共享服务器，实测 24 AICore/卡 @1800MHz） |
| 显存 | 64GB HBM/卡 |
| 基线探测时卡位 | AICore 0%（空闲），共享环境绝对值有 ±15µs 级扰动 |

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | openEuler (aarch64) 5.10.0-216 |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| CANN | 9.0.0（Ascend-cann-toolkit V100R001C10SPC001B250） |
| Triton | 3.5.1（ascend backend, BiShengIR 编译） |
| FlagGems | 5.3.5 |
| vllm | 0.20.2+empty（空壳，见§7） |

### 1.3 环境特殊性

- 共享 8 卡服务器；bench 数据在 0 卡空闲时段采集
- `flag_gems.enable()` 会接管 rand 族生成算子，其 rand kernel 在部分
  shape 下 UB 溢出编译失败——**测试输入一律 CPU 生成后搬 NPU**，
  绕开该已知缺陷（误归因教训见§4.3）

## 2. 算子定义

### 2.1 语义

```
out = softmax(Q·Kᵀ·scale + mask_bias) · V
  scale 缺省 1/√D
  is_causal: Sq==Skv 下三角遮蔽（与 aten 一致）
  attn_mask: bool → False 位 -inf；float → 加性 bias
  GQA: Hq≠Hkv 时 K/V 按 Hq/Hkv 组广播
```

### 2.2 接口签名

aten: `scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor`（4D）

### 2.3 数值规格

| 项 | 值 |
|---|---|
| 支持 dtype | fp16 / bf16 / fp32 |
| 内部精度 | softmax 状态与累加 fp32；QK/PV dot 输出 fp32 |
| fp32 dot 精度 | `input_precision="ieee"`（默认 tf32 的 10-bit 尾数实测 3.4e-4，extreme case 超差） |
| 容差 | fp32=1e-5 abs · fp16/bf16=2e-2 abs；extreme（softmax 饱和）用相对 5e-3 |
| 不支持 | dropout>0（确定性路径）、非 4D、D>512 |

### 2.4 shape 清单原则

必含非 BLOCK 整倍数序列（100/197/333——FlagOS-OP #15a 漏网教训的
固化）、D 非 2 幂（80）、GQA、Sq=1 decode 形态、全遮蔽行（NaN 语义）。

## 3. 实现说明

### 3.1 路线选择

宿主是标准 aten 算子 → A1。**但本栈注册点与常规认知不同**:
仅注册 PrivateUse1 **永远不被命中**——torch_npu 的 C++
AutogradPrivateUse1 包装（RegisterAutogradNPU.cpp:1000）不 redispatch，
直接调内部实现，把后端 key 覆盖短路（dispatch dump 中我们的注册显示
active 但 call count=0，实测证据链见 `probes/debug_reg*.py`）。正确拦截点是
**AutogradPrivateUse1**，且必须用 `autograd.Function` 包装（朴素函数
注册会破坏 autograd——输出无 grad_fn）。

### 3.2 kernel 设计

flash 风格 online-softmax two-pass（单 kernel）:

- Q 按 (BLOCK_M, D) 常驻；K/V 按 (BLOCK_N, D) 流式遍历
- running max/sum 在线归约，`alpha = exp(m_i − m_new)` rescale acc
- 尾块 masked load `other=-inf`，exp 后贡献 0——行级 online 归约
  不受 #15a 块内 tl.sum 污染影响（36 组含尾块 case 验证）
- causal **循环截断**: `hi = min(hi, cdiv((pid_m+1)·BLOCK_M, BLOCK_N))`
  （实测 1.93x 加速——原写法全量遍历+where 置-inf 白算上三角，
  编译器无法从运行时 mask 推断边界收缩）
- 固定 BLOCK 64×64（无 autotune，#15b；BM/BN>64 全组合实测 UB
  溢出编译失败）

### 3.3 本栈三条 Triton 硬约束（实测）

1. `tl.dot` 强制 lhs/rhs 同 dtype——P 矩阵须 cast 到 v.dtype，
   dot 的 acc 参数亦不可用（fp32 acc×alpha + fp16 v 不合法）
2. fp32 输入默认 tf32 精度（3.4e-4 级误差）——须显式 ieee
3. `tl.math.exp2` 慢路径（0.51x）——FA2 的 exp2 预折技巧在此栈
   负优化，保持 `tl.exp`

### 3.4 autograd

forward=Triton kernel；backward=标准 SDPA 数学梯度（fp32 ATen 组合，
与 reference 同层）: dV=PᵀdO · dP=dOVᵀ · dA=P⊙(dP−ΣdP⊙P) ·
dQ=dA·s·K · dK=dAᵀ·s·Q；GQA 梯度按组求和折叠回 Hkv。
梯度 vs 原生子进程对照: dq/dk 4.9e-4 · dv 3.9e-3（fp16 容差内）。

### 3.5 平台绑定与防误引（ascend910 专属）

本实现绑定 ascend910。三层防误引机制（均实测，`probes/guard_check.py`）:
模块元数据（`PLATFORM`/`SUPPORTED_DEVICE_TYPES`，供集成方程序化过滤）·
`sdpa_triton()` 调用守卫（非 npu tensor 立即 RuntimeError）·
`register_a1()` 注册守卫（无 torch_npu 环境注册 PrivateUse1 系 key
拦截）。绑定项 8 条清单与移植指引见 [PLATFORM.md](../PLATFORM.md)，
第二平台合并流程见 [MERGE.md](../MERGE.md)——函数名有意不改
（与 aten 对齐、入口稳定），平台区分在选择层做。

### 3.6 写法粒度变体实验（结论固化）

`script/perf_variants.py` 五变体单项 A/B（数值先验证后测性能）:

| 变体 | 结果 | 处置 |
|---|---|---|
| V4 causal 循环截断 | **1.93x 加速** | **已合入主 kernel** |
| V3 尾块分裂（完整块免 mask） | 0.96x 无收益 | 不合入（增加代码复杂度） |
| V2 exp2 预折（FA2 惯例） | 0.51x **负优化**（该栈 exp2 慢路径） | 不合入，PLATFORM.md §2 标注 |
| V1 转置布局加载（消 tl.trans） | 1.00x 无差异 | 不合入（编译期已消化） |
| V5 dot acc 融合 | 该栈语法不可用（acc 强制同 dtype） | 不合入 |

### 3.7 三级实现状态

| 级别 | 状态 | 说明 |
|---|---|---|
| torch 级 | ✅ | ATen 组合，kernel 层第二判卷人 |
| Triton 级 | ✅ | 主实现（本报告） |
| 硬件级 | ⬜ | 不开发——原生 CANN 闪电注意力已是该级最优参照，自研无超越空间（性能分册§3） |

## 4. 验证结果

### 4.1 结果矩阵

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 36 组精度（12 shape × 3 dtype）+ 哨兵 1 组，全过；最差 fp16 9.8e-4 / bf16 3.9e-3 / fp32 1.2e-7 |
| 框架层 | ✅ | A1 拦截命中（count>0）；注册路径 vs 直调**逐位一致**；vs 原生 1.2e-4；撤销恢复经独立进程验证 |
| 梯度 | ✅ | 三梯度 vs 原生子进程 ≤3.9e-3 |
| 黄金 | ✅ | 265 组（CPU fp32 生成，双向互验）triton 265/265 |
| 应用层 | ✅ | mini-decoder 消费方双跑: 拦截 28 · logits 1.95e-3 · 贪心续写一致率 1.00（§4.4） |

### 4.2 黄金生成的双向互验

expected 同时用 sdpa_reference 与 CPU `F.scaled_dot_product_attention`
(fp32) 计算，分歧超阈值即报错——防"参考恰好也是被测实现"式掩盖。
阈值按 dtype 分档（fp32 2e-5 / fp16 2e-2），extreme case 用相对阈值
（softmax 饱和下 fp32 累加顺序差异 ~3e-4 属正常，实测**原生实现与
黄金同为 3.381e-4**，两实现互咬合在 3.8e-6）。

### 4.3 应用层: mini-decoder 消费方双跑（轻量真实计算任务）

本栈 vllm 空壳，按 softmax-fullstack 先例构建 Python 层消费方:
Llama 风格 4 层 decoder（GQA 8/2 · causal · LayerNorm · MLP · fp16），
业务代码只调 `F.scaled_dot_product_attention`（aten 路径零改动），
基线/注册双跑同 seed 同输入:

| 断言 | 结果 |
|---|---|
| 拦截计数（4 层 × 7 forward） | 28 ✓ |
| logits max_diff | 1.95e-3（4 层深网络 fp16 容差内） |
| 全位置 top-1 一致率 | 1.00 |
| 贪心续写序列一致率 | 1.00（阈值 2/3，未用到余量） |

入口: `test/framework_level.py`。

### 4.4 开发过程中的三次误归因（记录为方法论）

1. mask 路径"编译失败" → 实为 flag_gems rand kernel UB 溢出（mask
   构造被接管），SDPA 本体正常——输入改 CPU 生成后消除
2. bool mask 误差 4.8e-2 → 实为参考写错（bool→float 转换把遮蔽语义
   变加性语义），kernel 无错——参考修正后逐 case 通过
3. extreme_fp16 误差 1.4e-1 → dtype 固有（输入量化翻转饱和 softmax
   的 argmax），**原生实现同误差**——该 case 改 fp32-only

## 5. 性能

三方对照（fp16/bf16 各 6 形状，详见[性能分册](performance.md)）:

| shape | 自研 | 原生 | FlagGems 直调 | 自研/原生 |
|---|---|---|---|---|
| prefill_1k_d128 | 0.39ms | 0.12ms | 6.6ms | 3.3x |
| prefill_2k_d128 | 1.44ms | 0.22ms | 25.0ms | 6.5x |
| prefill_4k_d128 | 5.52ms | 0.51ms | 97.3ms | 10.8x |
| gqa_1k_d128 | 0.76ms | 0.15ms | 13.0ms | 5.1x |
| decode_d128 | 0.10ms | 0.05ms | 0.13ms | 1.9x |

vs FlagGems: **3.3-17.6x 加速**。vs 原生差距的根因分解见性能分册
（栈 GEMM 效率 ~5.2x × 结构放大 ~2x，纯 matmul 天花板实验实证）。

## 6. 风险与已知边界

| # | 项 | 影响 | 缓解 |
|---|---|---|---|
| 1 | torch.library 注册进程内不可撤销 | 长驻进程混跑多实现需子进程隔离 | op 层测试已用子进程；文档标注 |
| 2 | 大 BM/BN 编译失败（UB 192KB） | tile 调优空间封死在 64×64 | 性能分册记录；待 BiShengIR 演进 |
| 3 | 共享服务器 bench 扰动 | 绝对值 ±15µs 级 | 相对比值口径；空闲时段采集 |
| 4 | Sq≠Skv 的 causal 语义未实现（aten 本身限制性语义） | 极端形状用例 | wrapper 断言拒绝，与参考一致 |

## 7. 结论与路线

Triton 级 SDPA 在 Ascend 910 上交付: 两层验证全绿、黄金 265/265、
比生态现有 Triton 实现快一个数量级、A1 拦截机制查明了 torch_npu
栈的正确注册点（AutogradPrivateUse1，有实验证据链）。

后续路线（按性价比）:

1. **上游 PR**: 本 kernel + NPU 接入修复贡献 FlagGems（dot-dtype/
   ieee/UB 坑文档化），精度/性能数据即 PR 证据；多平台结构对齐
   上游 `runtime/backend/_<vendor>/` 惯例（[MERGE.md](../MERGE.md) §4）
2. **PyPTO 重写**（需 910B/C + 新 CANN）: tile 级 DSL 暴露 L0/L1/UB
   与 multi-buffer 控制，预期吃到流水收益；pto-kernels 以 torch
   算子形式发布可平移本 A1 注册；但非 Triton DSL，进 FlagGems 主集
   需架构讨论
3. **第二平台扩展**: 按 [MERGE.md](../MERGE.md) 七步合入——黄金
   265 组与 kernel 层 36 case 直接复用，只写 kernel 本体 + §2 八项
   绑定重验
4. **真实 vLLM 补验**: 应用层已以 mini-decoder 双跑收口；待本栈
   有可用推理引擎后平移注入（A1 经 sitecustomize 进子进程）
