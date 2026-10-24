# 平台绑定清单（Platform Binding）

> 当前交付包含四个平台：**ascend910**（Triton 主实现）、
> **p800-kunlunxin**（厂商 efficient-attention 委托 + fp32 精度补偿）、
> **mlu590**（Cambricon TMO FA + fused overrideable 委托）与
> **thead-ppu**（平头哥 ZW810E：flash/efficient 分流委托 + fp32 组合）。
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

## 3b. mlu590 绑定（Cambricon fused/TMO 委托）

环境锁定：Torch `2.11.0+cpu` + torch_mlu `1.33.1`、Triton `3.4.0`、
FlagGems `5.3.5`、设备 **MLU590-M9 / `mlu:0`**。

| 绑定项 | 结论 |
|---|---|
| 设备表现 | `device.type="mlu"`；守卫额外要求 `torch_mlu` 可导入 |
| 注册 key | **`AutogradPrivateUse1`**（与 torch_npu 同栈结论；`AutogradMLU`/`MLU` 亦命中，但 PrivateUse1 是 torch_mlu 后端 key） |
| 原生 F.sdpa 实际路径 | **`_scaled_dot_product_fused_attention_overrideable` → CNNL FA v2**（不是 math） |
| 主实现 P0 | 直调同一 fused overrideable——≈1.0x 原生，常与原生逐位一致；**不可**在 backend 内再调 `F.sdpa`（A1 注册后递归） |
| 快路径 P1 | `torch_mlu_ops.flash_attention`（CNNL SDPA v7，**BSHD** q/k/v，bias **BHSD** `(B,H,Sq,Skv)`）；半精度 prefill 实测 **1.21-1.33x** 原生；`SDPA_MLU_TMO=0` 关闭 |
| 不可用 private 入口 | efficient/flash/cudnn attention 三个 aten op 在 MLU 上 CPU fallback 失败 |
| bool / GQA | bool→additive bias；overrideable 无 `enable_gqa` → `repeat_interleave` 扩 KV |
| causal+mask | 与 `_native_shim`/`F.sdpa` 相同：先折叠进 mask |
| 全遮蔽行 | fused/TMO 返回有限值，输出端 `masked_fill` 恢复 NaN |
| FlagGems `_cambricon` | 精度 397/397 全过，但比原生慢 10-40x 且无 autograd——仅作兜底/对照 |
| 性能水位 | **1.00-1.33x** 原生（早期 math-only 为 0.12-0.52x，已废） |

## 3c. thead-ppu 绑定（平头哥 ZW810E，flash/efficient 分流委托）

环境锁定：开发容器 `flagos-ppu-wt-dev`（torch **2.10.0** CUDA 兼容层 /
triton **3.6.0** / vllm 0.19.0；无 flag_gems、无 vllm-plugin-fl），
16 卡均为 **PPU-ZW810E**，用卡 **6,7**（避 4/5/11）。profile 显式
`--device thead-ppu`（`auto_detect.enabled=false`），测试设备默认
`cuda:6`，`SDPA_TEST_DEVICE` 可覆盖。

| 绑定项 | 结论 |
|---|---|
| 设备表现 | PPU 经 CUDA 兼容层直曝 `device.type="cuda"`；守卫要求设备名含 `PPU`（triton `PPUDriver` 为二路证据，见 `probes/thead_triton_dot_smoke.py`）。`get_impl("cuda")` 三候选路由：`torch_xmlir`→p800、PPU 栈→thead_ppu、否则报错 |
| 注册 key | **`AutogradCUDA`**。实测（`probes/thead_reg_probe.py`，每 key 独立进程）：`AutogradCUDA` 6/7 探针命中（唯一零命中是 `inference_mode`，同 known-issues #19 的 CUDA 栈现象）；纯 `CUDA` 7/7 全命中但反传出现"未注册 Autograd kernel"警告——故选 `AutogradCUDA`。注意 `configs/devices/thead-ppu.yaml` 的 `dispatch_key` 字段写的是 `CUDA`（P0 首测时的保守值），`ops/sdpa/_profile.py` 以 `_is_ppu_stack()` 检测覆盖为 `AutogradCUDA`，以本表为准 |
| GQA | efficient/flash schema **均无 `enable_gqa`**（8 参数实测）→ `repeat_interleave` 扩 KV |
| 主实现（maskless） | 直调 `aten::_scaled_dot_product_flash_attention`。`TorchDispatchMode` 实证 PPU 原生 `F.sdpa` 内部恰好只调这一个 overload——直调即与原生同 kernel；flash schema **无 attn_mask**（传入报 `Unknown keyword argument 'attn_mask'`） |
| 主实现（带 mask） | efficient 委托 `aten::_scaled_dot_product_efficient_attention`；bool mask 转 additive `-inf` bias；全遮蔽行输出端 `masked_fill` 恢复 NaN |
| bias 16B stride 对齐 | efficient kernel 强制要求。报错原文两种：`(stride(1)*elem)%16!=0` → `attn_bias is not correctly aligned (strideH)`；`(stride(2)*elem)%16!=0` → `attn_bias is not correctly aligned`（无后缀）。修复（`_aligned_bias`，探针里叫 kv_pad）：把 bias 复制进按列 pad 到 16B 步长的缓冲再 `as_strided` 回原逻辑形状（K-stride 保持 1）。**生效范围：fp32/fp16 已验证等价（`probes/thead_bias_layout.json` 9/9）；bf16 2026-10-24 已补验收口（`probes/out/thead_bf16_bias_pad.json`，T4）：实测生产 `_aligned_bias` vs 连续 kv_pad 布局直算 vs CPU fp32 参考 11/11 用例在 bf16 1e-2 abs（黄金口径收紧）内一致（两路径互差 0），报错原文两种变体在 bf16 均复现。黄金集 bf16 masked 案例全为 basic S128（row stride 256B %16==0），走连续 bias 不触发非对齐分支——黄金 24 组 masked bf16 子集经生产 facade 回归无回退（worst 7.8125e-3 = 交付时历史 worst）** |
| fp32 路径 | 委托精度本身很好（非 extreme 3e-7～5e-6），但 scale=8 饱和（黄金 extreme 口径）abs 2.79e-3 超 1e-5 判卷线 → **fp32 走 ATen 组合**（`sdpa_reference` 设备端）。PPU **无** p800 的 XMLIR bmm JIT 缺陷：S∈(320,640] 设备组合 vs CPU 参考 ≤5.7e-7，**无需 768 pad** |
| bf16 boundary | V≈dtype_max/2 × 长序列时厂商 kernel 中间量溢出为 inf（随 Sq 增大阈值下降）——**原生 `F.sdpa` 同样溢出**，属平台分化而非实现缺陷。tail 测试判定口径：与原生 `inf` 位置 parity + 有限子集 rel<5e-3（黄金 extreme 同款相对容差） |
| cudnn / overrideable | `_scaled_dot_product_cudnn_attention` 直调报 `CUDNN_BACKEND_TENSOR_DESCRIPTOR: ... ACDNN_STATUS_BAD_PARAM`；`_scaled_dot_product_fused_attention_overrideable` 报 `NotImplementedError`（该入口是 privateuse1 专用）——均不选 |
| triton 备选 | triton 3.6.0 在 PPU 可编译执行（target cuda arch=80, PPUDriver），但 `tl.dot` 精度掉到 fp16 量级（64³ 点积 rel≈7e-4；ieee K=128 abs 1.03e-4），仍超黄金 fp32 1e-5 → 仅记备选，不接生产 |
| 性能 | 首版全量走 efficient + `get_impl` 每调用 `find_spec` 解析 → **4.07× 倒挂**（0.426 ms vs 原生 0.105 ms）。收口两步：① maskless 分流 flash；② `backends.get_impl` 加进程级解析缓存（cProfile 实测 find_spec/importlib ~80-100 µs/调用）。最终 perf_run 台账 **0.114 ms vs 原生 0.104 ms（1.10×，37.7 TFLOPS）**，六形状 bench 0.63-0.99×（见 `reports/performance.md`） |
| 跨层一致性缺证 | 仓库级 `run.py --consistency` 锚定 gelu_and_mul（A2 路由），硬依赖 `flag_gems` + `vllm_fl`——dev 容器二者均无（`ModuleNotFoundError: flag_gems`），属**结构性缺证**而非 sdpa/thead 缺陷。sdpa 的"直调↔dispatch 张量级一致"由 `test/op_level.py` 第 3 步逐位断言覆盖（`bitwise_vs_direct=True`） |

## 4. 平台专属数字（不可跨平台引用）

- ascend910 性能与根因分解见 `reports/performance.md`、
  `reports/perf_analysis.md`。
- p800-kunlunxin 性能采样见 `reports/perf_fp16_p800-kunlunxin.json`
  （每次读取一个输出元素强制完成；P800 委托路径相对 Python
  `F.sdpa` 加速 1.02-1.14x，FlagGems Triton 在 2k/4k 序列比 ours
  慢 5.1-6.6x）。
- “FlagGems 未被 aten 分发接管”是 torch_npu 栈特定结论；P800 上
  FlagGems `_kunlunxin` attention 可直调，但 mask 场景存在 launch 失败。
- mlu590 性能与路径选择见 `reports/mlu590.md`、
  `reports/perf_fp16_mlu590.json`（TMO+fused 后 **1.00-1.33x** 原生；
  FlagGems 慢 10-40x，仅兜底）。
- thead-ppu 性能与倒挂收口见 `reports/thead-ppu.md`、
  `reports/performance.md`、`reports/perf_fp16_thead-ppu.json`
  （maskless flash 分流 + `get_impl` 解析缓存后 **1.10x** 原生，
  六形状 0.63-0.99x；FlagGems 对比因 dev 栈缺失记缺证）。

## 5. 多平台结构（已落地）

```
kernel/triton_level.py          # 兼容 facade：按 query.device.type 分发
kernel/backends/
  ascend910.py                  # 原 Triton 主体，旧功能零改动
  p800_kunlunxin.py             # Kunlunxin 厂商委托 + fp32 补偿
  mlu590.py                     # Cambricon TMO FA + fused overrideable
  thead_ppu.py                  # 平头哥 flash(maskless)/efficient(masked) 分流委托
```

合并流程与后续平台扩展仍按 [MERGE.md](MERGE.md) 执行。

多算子同进程/perf registry 会按 provider 顺序加载多个 `reference.py`。
backend 内不允许裸 `from reference import ...` 作为首选路径：一旦其他
算子的顶层 `reference` 已进入 `sys.modules`，该导入会解析到错误模块。
正确顺序是包态相对导入优先（如 `from ...reference import ...`），仅
单算子脚本模式 fallback 到顶层 `reference`。该问题已在
`tests/unit/test_sdpa_backend_import.py` 固化，实测记录见 PR #22。

## 6. 引用防误用

函数名**有意**保持 `sdpa_triton`（公共 API 稳定）。平台区分靠三层：

| 层 | 机制 |
|---|---|
| facade 元数据 | `kernel.triton_level.PLATFORM = "multi(...)"`、`SUPPORTED_DEVICE_TYPES=("npu","cuda","mlu")` |
| backend 元数据 | `kernel.backends.ascend910.PLATFORM` / `...p800_kunlunxin.PLATFORM` / `...mlu590.PLATFORM` / `...thead_ppu.PLATFORM` |
| 调用守卫 | backend 只接受本设备且 P800 要求 XMLIR、MLU 要求 torch_mlu、thead 要求设备名含 `PPU`；误配立即 `RuntimeError` 并指向本文件 |

```bash
# 每个平台各跑一次
python3 probes/guard_check.py ascend910
python3 probes/guard_check.py p800-kunlunxin
python3 probes/guard_check.py mlu590
python3 probes/guard_check.py thead-ppu
```
