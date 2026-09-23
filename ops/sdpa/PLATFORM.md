# 平台绑定清单（Platform Binding）

> 当前交付包含两个平台：**ascend910**（Triton 主实现）与
> **p800-kunlunxin**（厂商 efficient-attention 委托 + fp32 精度补偿）。
> 本文件记录不可凭直觉搬运的绑定结论；跨平台移植时逐项重验，不要把
> 单平台实验结果当成普适行为。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| fp32 语义参考 | `reference.py` |
| 黄金规格与 397 组数据 | `goldendata/`（CPU 生成，天然跨平台） |
| 公共入口与平台选择器 | `kernel/triton_level.py` → `kernel/backends/` |
| 三层测试、哨兵与性能脚本 | `test/`、`script/`（传 profile/device 即可） |
| A1 autograd.Function 包装与数学 backward | `register.py` |

旧引用 `from kernel.triton_level import sdpa_triton` 保持不变；实现主体
已按平台拆到 `kernel/backends/`。

## 2. ascend910 绑定（Triton 实现）

| 绑定项 | 位置 | 实测结论 |
|---|---|---|
| 注册 key = `AutogradPrivateUse1` | `register.py` | torch_npu C++ 包装不 redispatch；仅 `PrivateUse1` 永不命中 |
| `tl.dot` 前同 dtype cast | `kernel/backends/ascend910.py` | Ascend 语义层要求 lhs==rhs |
| fp32 `input_precision="ieee"` | 同上 | 默认 tf32 在 extreme case 超 1e-5 容差 |
| 保持 `tl.exp` 而非 exp2 | 同上 | exp2 在 triton-ascend 是 0.51x 慢路径 |
| BLOCK 64×64 固定、不 autotune | 同上 | 更大 tile 触发 UB 192KB 溢出；autotune 可选非法 warps |
| causal 循环截断 | 同上 | S=2048 causal 1.93x 收益 |
| device 上下文包启动 | 同上 | 防 FlagOS-OP #11 静默 no-op |
| causal+mask 折叠 | `kernel/_native_shim.py` | torch_npu aten 拒绝二者并存 |

## 3. p800-kunlunxin 绑定（厂商委托实现）

环境锁定：Torch `2.9.0+cu129`、`torch_xmlir`
`XMLIR--bc1b1dc6f-dev+2026032411`、Triton `3.0.0`，
`CUDA_VISIBLE_DEVICES=1,2`。本机稳定测试设备是 **`cuda:1`**；`cuda:0`
上厂商 SDPA 曾出现挂起，不要把该现象误判为本实现缺陷。

| 绑定项 | 结论 |
|---|---|
| 设备表现 | XMLIR 将 XPU tensor 暴露为 `device.type="cuda"`；调用守卫额外要求 `torch_xmlir` 可导入，避免误接 NVIDIA CUDA |
| 注册 key | **`AutogradCUDA`**。该 key 下 Python wrapper 会收到省略默认值的 schema 参数，必须在注册层补齐后再进 `autograd.Function` |
| 主实现 | 直调 `aten::_scaled_dot_product_efficient_attention`，绕过 Python `F.sdpa` monkey patch，避免 auto 分发递归；bool mask 转 additive `-inf` bias |
| autograd | q/k/v 直调反向必须要求 forward 计算 log-sumexp；float mask 反向厂商暂不支持 `bias_requires_grad`，backend 复用 A1 数学 backward |
| fp32 路径 | 厂商 kernel 对 fp32 的误差约 2e-4，超过黄金 1e-5；改走 fp32 ATen 组合，实测误差 ≤7e-7 |
| XMLIR bmm bug | `S∈(320,640]` 的组合 bmm 可触发 `bmm_one_loop` JIT 编译错误；fp32 路径将序列 pad 到 768 并用合法 mask 还原语义 |
| 应用层 dtype | 本随机 mini-decoder 的 Linear/LayerNorm fp16 链路溢出；P800 使用 bf16，输出与原生逐位一致 |

## 4. 平台专属数字（不可跨平台引用）

- ascend910 性能与根因分解见 `reports/performance.md`、
  `reports/perf_analysis.md`。
- p800-kunlunxin 性能采样见 `reports/perf_fp16_p800-kunlunxin.json`
  （每次读取一个输出元素强制完成；P800 委托路径相对 Python
  `F.sdpa` 加速 1.02-1.14x，FlagGems Triton 在 2k/4k 序列比 ours
  慢 5.1-6.6x）。
- “FlagGems 未被 aten 分发接管”是 torch_npu 栈特定结论；P800 上
  FlagGems `_kunlunxin` attention 可直调，但 mask 场景存在 launch 失败。

## 5. 多平台结构（已落地）

```
kernel/triton_level.py          # 兼容 facade：按 query.device.type 分发
kernel/backends/
  ascend910.py                  # 原 Triton 主体，旧功能零改动
  p800_kunlunxin.py             # Kunlunxin 厂商委托 + fp32 补偿
```

合并流程与后续平台扩展仍按 [MERGE.md](MERGE.md) 执行。

## 6. 引用防误用

函数名**有意**保持 `sdpa_triton`（公共 API 稳定）。平台区分靠三层：

| 层 | 机制 |
|---|---|
| facade 元数据 | `kernel.triton_level.PLATFORM = "multi(...)"`、`SUPPORTED_DEVICE_TYPES=("npu","cuda")` |
| backend 元数据 | `kernel.backends.ascend910.PLATFORM` / `kernel.backends.p800_kunlunxin.PLATFORM` |
| 调用守卫 | backend 只接受本设备且 P800 要求 XMLIR 栈；误配立即 `RuntimeError` 并指向本文件 |

```bash
# 每个平台各跑一次
python3 probes/guard_check.py ascend910
python3 probes/guard_check.py p800-kunlunxin
```
