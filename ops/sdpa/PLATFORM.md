# 平台绑定清单（Platform Binding）

> 本目录当前是 **ascend910 专属交付**（profile: `ascend910`，
> Ascend 910_9382 / CANN 9.0.0 / triton-ascend 3.5.1）。
> 将来移植到其他平台时，按本清单核对——**不要**把列出的
> Ascend 绑定结论当成普适行为。

## 1. 可直接移植（平台无关，语义层）

| 内容 | 位置 |
|---|---|
| 语义参考（fp32 SDPA 数学） | `reference.py` |
| 黄金规格与 265 组数据 | `goldendata/`（CPU 生成，天然跨平台） |
| kernel 主体结构（online-softmax two-pass、GQA/causal/mask 语义、尾块 masked load） | `kernel/triton_level.py` |
| 测试三层与哨兵 | `test/`（dev 改 profile 即可） |
| A1 注册的 autograd.Function 形态 | `register.py` |

## 2. Ascend 绑定（移植时必须逐项重验）

| 绑定项 | 位置 | 绑定原因（实测） | 其他平台预期 |
|---|---|---|---|
| 注册 key = `AutogradPrivateUse1` | `register.py` | torch_npu 的 C++ Autograd 包装不 redispatch，PrivateUse1 永不命中 | CUDA 用 `CUDA` key 即可（FlagOS-OP 模板默认形态） |
| `tl.dot` 前同 dtype cast | `kernel/triton_level.py` P-cast | Ascend 语义层 assert lhs==rhs | CUDA 无此限制（可去掉 cast） |
| fp32 用 `input_precision="ieee"` | 同上 | 默认 tf32 在 extreme 实测 3.4e-4 超差 | CUDA 同样建议 ieee（行为一致，动机相同） |
| **不用 exp2**（保持 tl.exp） | 同上 | `tl.math.exp2` 在 triton-ascend 是慢路径（0.51x） | **CUDA 上 exp2 是正优化**——移植时应改回 FA2 惯例 |
| BLOCK 64×64 固定、不 autotune | 同上 | BM/BN>64 全组合 UB(192KB) 溢出编译失败；autotune 会选非法 num_warps | CUDA 应重新调优（128/64 tile + autotune 通常更优） |
| causal 循环截断 | 同上 | 1.93x 实测收益 | 平台无关的正优化（机制通用），保留 |
| device 上下文包启动 | 同上 | FlagOS-OP #11 静默 no-op | flag_gems 栈通用；原生 Triton 可不需要 |
| `kernel/_native_shim.py` 因果折叠 | `_native_shim.py` | NPU aten 拒绝 causal+mask 并存 | CUDA aten 接受并存，shim 可退化为直通 |

## 3. 平台专属数字（不可跨平台引用）

- 性能：三方对照全部数字（原生=CANN 闪电注意力，仅存在于本栈）
- 根因分解（GEMM 栈 5.2x / UB tile 上限）：`reports/perf_analysis.md`
- FlagGems "未被 aten 分发接入"：torch_npu 注册机制所致，平台特定

## 4. 多平台演化预案（将来真有第二平台时）

**具体操作指南已就位: [MERGE.md](MERGE.md)**（七步流程 + 平台选择器
+ FlagGems 上游路径 + 完成判据）。骨架:

```
kernel/triton_level.py          # 退化为 facade（sdpa_triton 入口不变）
kernel/backends/                # 第二平台出现时才建（MERGE.md §1）
  ascend910.py                  #   现 triton_level.py 主体移入
  <platform2>.py
configs/devices/<platform>.yaml # 设备 profile（对齐 FlagOS-OP）
```

判据：当第二平台的 §2 清单开始与 ascend910 分叉（而非只改参数），
才拆 backends/；此前参数级差异用 profile 字段承载。

## 5. 引用防误用（框架集成方必读）

函数名**有意**保持 `sdpa_triton`（与 aten 算子对齐、入口稳定），
平台区分靠以下三层机制，而非改名：

| 层 | 机制 | 位置 |
|---|---|---|
| 元数据 | `kernel.triton_level.PLATFORM = "ascend910"` / `SUPPORTED_DEVICE_TYPES = ("npu",)` | 集成方程序化过滤 |
| 调用守卫 | 非 npu tensor 调用立即 `RuntimeError`（指回本文件） | `sdpa_triton()` 入口 |
| 注册守卫 | 无 torch_npu 环境注册 PrivateUse1 系 key 立即 `RuntimeError` | `register_a1()` |

框架侧的选择器写法示例（多平台 dispatch 时）：

```python
from kernel import triton_level
impls = {"ascend910": triton_level.sdpa_triton}   # 按 PLATFORM 注册
# 第二平台加入时在此扩展，误配平台会在调用守卫处显式失败
```
