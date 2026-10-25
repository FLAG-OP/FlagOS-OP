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
| 注册 key | **`AutogradCUDA`**（算子本地注册点）。实测（`probes/thead_reg_probe.py`，每 key 独立进程）：`AutogradCUDA` 6/7 探针命中（唯一零命中是 `inference_mode`，同 known-issues #19 的 CUDA 栈现象）；纯 `CUDA` 7/7 全命中但反传出现"未注册 Autograd kernel"警告——故选 `AutogradCUDA`。中央 YAML 另存基键 `CUDA`，与本行**有意分层**并存（非笔误），依据见 §3c-1 |
| GQA | efficient/flash schema **均无 `enable_gqa`**（8 参数实测）→ `repeat_interleave` 扩 KV |
| 主实现（maskless） | 直调 `aten::_scaled_dot_product_flash_attention`。`TorchDispatchMode` 实证 PPU 原生 `F.sdpa` 内部恰好只调这一个 overload——直调即与原生同 kernel；flash schema **无 attn_mask**（传入报 `Unknown keyword argument 'attn_mask'`） |
| 主实现（带 mask） | efficient 委托 `aten::_scaled_dot_product_efficient_attention`；bool mask 转 additive `-inf` bias；全遮蔽行输出端 `masked_fill` 恢复 NaN |
| 非方阵 causal（`Sq≠Skv` 且 `is_causal=True`） | **2026-10-25 PR36 条目 4 接入**：不进 maskless flash 分支（flash 直调对该输入**不生效**，6 形状 × fp16/bf16 err 1.2-1.9），强制 efficient 委托——实测 6 形状 × 3 dtype err ≤5.63e-4(fp16)/≤2.98e-3(bf16)/≤2.54e-7(fp32)，与原生 `F.sdpa` 同容差且逐案同值；原 `ValueError("causal 语义要求 Sq == Skv...")` 已删。联动约束同带 mask 路径：LSE 按 32 倍数 pad（`lse[..., :Sq]` 切回 err ≤9.5e-7）、非对齐 bias 走 `_aligned_bias`（3/3，其中 2 例 `bias_repaired=True`）、GQA `repeat_interleave`（4/4）、厂商 efficient 反向可用（dq/dk/dv ≤3.9e-3）。证据 `probes/out/thead_nonsq_causal_probe.json`（现行）+ `..._prefix.json`（整改前快照）；**梯度侧同批拆门**：`register.py` 数学 backward 原 `if ctx.is_causal and Sq == Skv` 在非方阵上静默不折 causal（实测梯度差 4.68），已改为一律按 `(Sq, Skv)` 左上 tril |
| bias 16B stride 对齐 | efficient kernel 强制要求。报错原文两种：`(stride(1)*elem)%16!=0` → `attn_bias is not correctly aligned (strideH)`；`(stride(2)*elem)%16!=0` → `attn_bias is not correctly aligned`（无后缀）。修复（`_aligned_bias`，探针里叫 kv_pad）：把 bias 复制进按列 pad 到 16B 步长的缓冲再 `as_strided` 回原逻辑形状（K-stride 保持 1）。**生效范围：fp32/fp16 已验证等价（`probes/thead_bias_layout.json` 9/9）；bf16 2026-10-24 已补验收口（`probes/out/thead_bf16_bias_pad.json`，T4）：实测生产 `_aligned_bias` vs 连续 kv_pad 布局直算 vs CPU fp32 参考 11/11 用例在 bf16 1e-2 abs（黄金口径收紧）内一致（两路径互差 0），报错原文两种变体在 bf16 均复现。黄金集 bf16 masked 案例全为 basic S128（row stride 256B %16==0），走连续 bias 不触发非对齐分支——黄金 24 组 masked bf16 子集经生产 facade 回归无回退（worst 7.8125e-3 = 交付时历史 worst）** |
| fp32 路径 | 委托精度本身很好（非 extreme 3e-7～5e-6），但 scale=8 饱和（黄金 extreme 口径）abs 2.79e-3 超 1e-5 判卷线 → **fp32 走 ATen 组合**（`sdpa_reference` 设备端）。PPU **无** p800 的 XMLIR bmm JIT 缺陷：S∈(320,640] 设备组合 vs CPU 参考 ≤5.7e-7，**无需 768 pad** |
| bf16 boundary | V≈dtype_max/2 × 长序列时厂商 kernel 中间量溢出为 inf（随 Sq 增大阈值下降）——**原生 `F.sdpa` 同样溢出**，属平台分化而非实现缺陷。tail 测试判定口径：与原生 `inf` 位置 parity + 有限子集 rel<5e-3（黄金 extreme 同款相对容差） |
| cudnn / overrideable | `_scaled_dot_product_cudnn_attention` 直调报 `CUDNN_BACKEND_TENSOR_DESCRIPTOR: ... ACDNN_STATUS_BAD_PARAM`；`_scaled_dot_product_fused_attention_overrideable` 报 `NotImplementedError`（该入口是 privateuse1 专用）——均不选 |
| triton 备选 | triton 3.6.0 在 PPU 可编译执行（target cuda arch=80, PPUDriver），但 `tl.dot` 精度掉到 fp16 量级（64³ 点积 rel≈7e-4；ieee K=128 abs 1.03e-4），仍超黄金 fp32 1e-5 → 仅记备选，不接生产 |
| 性能 | 首版全量走 efficient + `get_impl` 每调用 `find_spec` 解析 → **4.07× 倒挂**（0.426 ms vs 原生 0.105 ms）。收口两步：① maskless 分流 flash；② `backends.get_impl` 加进程级解析缓存（cProfile 实测 find_spec/importlib ~80-100 µs/调用）。最终 perf_run 台账 **0.114 ms vs 原生 0.104 ms（1.10×，37.7 TFLOPS）**，六形状 bench 0.63-0.99×（见 `reports/performance.md`） |
| 跨层一致性缺证 | 仓库级 `run.py --consistency` 锚定 gelu_and_mul（A2 路由），硬依赖 `flag_gems` + `vllm_fl`——dev 容器二者均无（`ModuleNotFoundError: flag_gems`），属**结构性缺证**而非 sdpa/thead 缺陷。sdpa 的"直调↔dispatch 张量级一致"由 `test/op_level.py` 第 3 步逐位断言覆盖（`bitwise_vs_direct=True`） |

### 3c-1. 中央 YAML `dispatch_key` 与算子本地 key 的分层（PR36 条目 2 取证，2026-10-25）

`configs/devices/thead-ppu.yaml` 的 `device.dispatch_key: CUDA` 是**有意保留的中央
基键**，与上表算子本地 `AutogradCUDA` 分层并存，不是笔误：

| 侧 | key | 消费点（取证到的读取路径） |
|---|---|---|
| 中央 | `CUDA`（基键） | `common/device.py` → `tests/framework_level/_harness.py` / `_aten_harness.py` 写 `A1_DISPATCH_KEY` → `injection/sitecustomize.py` 注册 `aten::silu`；`scripts/intake_validate.py`、`scripts/bench_dispatch.py`、`tests/op_level/test_a1.py` 同读该字段 |
| 算子本地 | `AutogradCUDA` | `ops/<op>/_profile.py` → op / kernel / framework 三层测试与探针 |

中央取基键而非 `AutogradCUDA` 的依据（2026-10-25 卡 6 实测，`python3
scripts/smoke_dispatch_key.py` 复现，三断言全绿）：

- 基键 `CUDA` 在 grad / no_grad / inference_mode 三口径各拦截 1 次、输出逐位相同，
  反向梯度有限且对齐 silu 解析反向（`max_err=1.2e-7`）；
- `AutogradCUDA` 在 `inference_mode` 零命中（known-issues #19.3，即本节上表 6/7 的
  唯一零命中）——中央 A1 注册若改用它会在 inference 上静默失拦；
- 先例一致：`p800-kunlunxin{,-gems5}.yaml` = `CUDA`、`cambricon.yaml` =
  `PrivateUse1`，同为「YAML 基键 / 算子本地 `Autograd*` 键」分层（cambricon 侧
  分层已由 `ops/embedding/reports/cambricon.md` §4 记录）。

算子本地 `AutogradCUDA` 的探针依据不变（上表「注册 key」行）；两键分层不改任何
既有测试口径。

### 3c-2. 设备编号口径（PR36 条目 3 取证，2026-10-25）

审阅质疑「profile 默认 `cuda:6` 与 `visible_devices: "6,7"` 并存，经
`CUDA_VISIBLE_DEVICES` 重映射后子进程合法索引可能只有 `cuda:0/1`」。取证结论：
`configs/devices/thead-ppu.yaml` 与本算子本地 `cuda:6` **都取物理卡号**，
kernel / op / framework 三层同一口径，字段值不改。

| 层 | 入口（消费点） | 实际用卡 |
|---|---|---|
| kernel | 中央 YAML `device.torch_device` → `tests/kernel_level/*`；本地 `_profile.py` thead 节 → `ops/<op>/test/kernel_level*.py` | 物理卡 6 |
| op | 中央 YAML → `tests/op_level/*`；本地 `_profile.py` → `ops/<op>/test/op_level.py`、探针 | 物理卡 6 |
| framework（算子应用层） | 同读 `profile.torch_device` → `ops/<op>/test/framework_level.py` | 物理卡 6 |
| framework（中央 vLLM harness） | YAML `device.visible_devices` → `tests/framework_level/_harness.py:62` 写子进程 `CUDA_VISIBLE_DEVICES` | 物理卡 6,7（子进程逻辑 0/1） |

依据：

- 全仓只有三处写 `CUDA_VISIBLE_DEVICES`：`tests/framework_level/_harness.py:62`、
  `scripts/drift_study.py:31`、`examples/b-fullstack/example.py:208`。三处都只写
  **vLLM 子进程** env（`os.environ.copy()` 的副本），父进程与 kernel/op 算子进程
  从不重映射——仓库内没有物理→逻辑的映射逻辑可依，所以 profile 只能、也只需
  按物理卡号声明。
- 被重映射的子进程是 `tests/framework_level/_vllm_runner.py`，它只消费
  `profile.engine_args()` 与 `profile.name`，**不读 `profile.torch_device`**——
  `cuda:6` 不会在重映射子进程里被使用，仓库驱动流程因此没有越界路径。
  `examples/b-fullstack/example.py` 的 `stage_l2`（用 `torch_device`）与
  `stage_l4`（写 CVD）分属不同阶段、不同进程环境，同理不冲突。
- 实测（2026-10-25，跑前 `ppu-smi` 确认 6/7 空闲）：无 CVD 时 `device_count=16`、
  `cuda:6` 可用；`CUDA_VISIBLE_DEVICES=6,7` 的子进程 `device_count=2`、
  `get_device_name(0)="PPU-ZW810E"`，此时 `cuda:6` 报 `invalid device
  ordinal`——即**只有外层 shell 自行 `export` CVD 时**才需换算：物理卡 =
  列表第 N 项，`cuda:0/1` = 物理 `6/7`（换算说明沿用
  `docs/device-profiles.md`「昆仑芯卡号与入口选择」表）。
- 先例比对：`p800-kunlunxin{,-gems5}` = `cuda:1` + `"1,2"`、
  `cambricon`/`mlu590` = `mlu:0` + `"0"`——同为「`torch_device` 索引 ∈
  `visible_devices` 集合」的物理卡号写法。

本节只补口径声明与换算规则，未改任何字段值、未放宽任何既有断言。

### 3c-3. 非方阵 causal（Sq≠Skv）四平台现状（PR36 条目 4 取证，2026-10-25）

审阅质疑「T-Head SDPA 文档写明 Sq != Skv 的 causal 会转 efficient，但当前
backend 仍直接拒绝该输入」。取证三步：

**1) 拒绝发生在哪一行**：`kernel/backends/thead_ppu.py` 旧第 233-234 行
（dtype/head_dim 校验之后、flash/efficient 分流之前）
`if is_causal and query.shape[-2] != key.shape[-2]: raise ValueError("causal
语义要求 Sq == Skv（与 aten/reference 一致）")`；同款拒绝另见
`p800_kunlunxin.py:211`、`mlu590.py:272`；判卷侧 `reference.py` 旧第 60 行断言
`assert Sq == Skv`（现该断言已删）。

**2) 文档作出该路由结论的原文行**：`ops/sdpa/PLATFORM.md` 自身**没有**这条
结论——grep `Sq` / `Skv` / `causal` 只命中 §2 的 ascend 两行、§3b 的 mlu590
快路径行与 `causal+mask` 行、§3c 的 bf16 boundary 行（均为别平台、别语义）。
作出该结论的是同一 T-Head 交付里的 **sdpa_math 侧文档**：

| 文件:行 | 原文 |
|---|---|
| `ops/sdpa_math/PLATFORM.md:54` | 「主实现（带 mask 或**非方阵 causal**） → efficient 委托」 |
| `ops/sdpa_math/PLATFORM.md:55` | 「非方阵 causal **强制走 efficient**」（附首测 flash O err≈2.0 vs efficient err≈0） |
| `ops/sdpa_math/reports/thead-ppu.md:41` | 「带 mask 或**非方阵 causal → efficient**」 |
| `ops/sdpa_math/reports/development.md:77`、`ops/sdpa_math/REPORT.md:25` | 同款分流行 |

而 `ops/sdpa/reports/development.md` **改前第 214 行**（本批已改写为选型 a
的结论）反向写「Sq≠Skv 的 causal 语义未实现（**aten 本身限制性语义**）→
wrapper 断言拒绝，与参考一致」——sdpa 侧按拒绝、sdpa_math 侧按转
efficient，两边对不上，即审阅所指矛盾。

**3) 实测**（`probes/thead_nonsq_causal_probe.py`，跑卡前 `ppu-smi` 确认
6/7 空闲 2MiB/0%；产物：现行 `probes/out/thead_nonsq_causal_probe.json`
+ 整改前只读快照 `probes/out/thead_nonsq_causal_probe_prefix.json`）：

| 探针项 | 结果 |
|---|---|
| 生产 facade `sdpa_triton` | **整改前**（快照 `..._prefix.json`）：6 非方阵形状 × 3 dtype = 18 例全 `ValueError` + `reference` 1 例 `AssertionError`（`summary REJECT 19`），方阵对照 3 例 PASS；**整改后**（本 JSON）：21 例全 PASS，非方阵 18 例 err 0.000e+00 |
| 原生 `F.sdpa`（PPU） | 全 PASS——aten 语义是 **top-tril** `tril(ones(Sq, Skv))`，非方阵同样支持；旧 assert 的「与 aten 一致」不成立 |
| flash 直调 `is_causal=True` | 全 FAIL（err 1.21-1.91，causal 未施加）→ **不能**当非方阵 causal 路径 |
| efficient 直调 `is_causal=True` | 6 形状（4×8/8×4/1×64/100×197/333×197/64×1）× fp16/bf16/fp32 全 PASS：err ≤5.62e-4 / ≤2.98e-3 / ≤2.53e-7，与原生逐案同误差 |
| LSE 32-pad 切回 | `lse.shape[-1]` = Sq 上取整到 32（4→32、100→128、333→352），`[..., :Sq]` 后 err ≤9.5e-7 |
| `_aligned_bias` × 非方阵 causal | bool 333×197、float 100×197 PASS 且 `bias_repaired=True`（bias 确被修复）；bool 1×64 PASS `bias_repaired=False`（Sq=1 无需 pad） |
| GQA 扩 KV × 非方阵 causal | Hq/Hkv=8/2 的 1×64、100×197 PASS |
| 厂商 efficient 反向 | dq 2.4e-4 / dk 4.9e-4 / dv 3.9e-3（fp16 容差内） |
| A1 数学 backward（**旧门回放**，5b 内联复刻整改前门体） | **MISMATCH err=4.678**——非方阵上不折 causal；现门已拆，现行判卷见 `test/op_level.py` 第 6 步 |

**选型：a）实现按文档路由**。依据是探针而不是推断：efficient 委托六形状三
dtype 全对、与原生同值，且原生本就支持该输入（A1 注册不该比原生更严）；代价
不结构性——只需删拒绝行、把 maskless 分支的非方阵 causal 改道 efficient、
拆 `register.py:109` 梯度门、`reference.py` 与 aten 对齐（黄金集无非方阵
causal 案例，397 例判卷口径不变）。四平台现状：

| 平台 | 非方阵 causal（`is_causal=True` 且 `Sq≠Skv`） | 位置 |
|---|---|---|
| thead-ppu | **支持**：maskless 与带 mask 都走 efficient（fp16/bf16/fp32、GQA、非对齐 bias、反向全验） | `kernel/backends/thead_ppu.py` 分流 + `register.py` 梯度 |
| p800-kunlunxin | 显式拒绝 `ValueError`（本平台未接入，未在 PPU 之外实测） | `kernel/backends/p800_kunlunxin.py` |
| mlu590 | 显式拒绝 `ValueError`（同上） | `kernel/backends/mlu590.py` |
| ascend910 | 显式拒绝 `ValueError`——原为 `IS_CAUSAL=is_causal and Sq == Skv` **静默降级成不遮蔽**（前向错、且与拆门后的反向折 causal 不一致），本批补同款拒绝 | `kernel/backends/ascend910.py` |

配套测试：`test/kernel_level_tail_thead_ppu.py` 第 3 节（8 形状 × 3 dtype ×
2 seed + 与原生 parity 3 例）、`test/op_level.py` 第 6 步（thead 专用：A1 拦截、
注册=直调逐位、梯度 vs CPU 参考）。

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
