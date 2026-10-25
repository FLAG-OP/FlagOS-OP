# thead-ppu SDPA 移植与验证报告

日期：2026-10-23（含 flash/efficient 分流与性能倒挂收口）

## 1. 环境

| 项 | 值 |
|---|---|
| 容器 | `flagos-ppu-wt-dev`（flagtree-ppu-3.6-dev:202608，py3.12.3） |
| Torch | 2.10.0（CUDA 兼容层，16 卡 `cuda:0-15`） |
| Triton | 3.6.0（PPUDriver，target cuda arch=80） |
| vLLM | 0.19.0+cu130 |
| FlagGems / vllm-plugin-fl | **无**（dev 栈结构性缺失，见 §6.11） |
| 设备 | **PPU-ZW810E × 16**；用卡 **6,7**（避 4/5/11）；测试卡 `cuda:6` |
| profile | `thead-ppu`（`auto_detect.enabled=false`，显式 `--device thead-ppu`；`SDPA_TEST_DEVICE` 可覆盖测试卡） |

环境核对：

```bash
ppu-smi                 # PPU-ZW810E；确认 6/7 空闲
PYTHONPATH=/hosthome/FlagOS-OP python3 probes/guard_check.py thead-ppu
```

## 2. 实现选择

### 2.1 不可用 / 不优先路径（实测）

| 尝试 | 结果 |
|---|---|
| `aten::_scaled_dot_product_cudnn_attention` | `cuDNN Frontend error: CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_IS_BY_VALUE Failed cudnn_status: ACDNN_STATUS_BAD_PARAM` |
| `aten::_scaled_dot_product_fused_attention_overrideable` | `NotImplementedError`（privateuse1 专用入口，CUDA 栈无注册） |
| triton 3.6.0 ieee 自研 | 可编译可运行，但 `tl.dot` 精度掉到 fp16 量级（64³ 点积 rel≈7e-4；ieee K=128 abs 1.03e-4），超黄金 fp32 1e-5——仅记备选 |
| Python `F.sdpa` 作为 backend 内部调用 | A1 注册后经 dispatcher 递归（p800 同款教训） |
| fp16/bf16 全量走 efficient 委托（首版） | 精度全绿但 **4.07× 性能倒挂**（0.426 ms vs 原生 0.105 ms），已废（见 §5） |

### 2.2 最终路径（分层）

```
业务 F.sdpa → A1 AutogradCUDA
  → thead_ppu.sdpa_triton
       ① fp32 → ATen 组合（sdpa_reference 设备端；extreme 饱和补偿，
                无 p800 bmm bug、无需 768 pad）
       ② fp16/bf16 + maskless → flash 直调
          aten::_scaled_dot_product_flash_attention
       ③ fp16/bf16 + mask → efficient 委托
          aten::_scaled_dot_product_efficient_attention
          （bool→additive -inf bias；16B stride 不齐时 _aligned_bias
           列 pad + as_strided；全遮蔽行 masked_fill 恢复 NaN）
       ④ 可微 additive mask → _SDPA_A1_Function 数学 backward
```

- **GQA**：flash/efficient schema 均无 `enable_gqa`（8 参数实测）→
  `repeat_interleave` 扩 KV。
- **maskless 选 flash 的依据**：`TorchDispatchMode` 实证 PPU 原生
  `F.sdpa` 内部恰好只调 `_scaled_dot_product_flash_attention` 这一个
  overload——直调即与原生同 kernel；flash schema 无 attn_mask（传入报
  `Unknown keyword argument 'attn_mask'`），故带 mask 场景仍走 efficient。
- flash 输出连续；efficient 输出 D-major 非连续（`reshape(-1)` 会整份
  拷贝）——半精度委托路径的隐性开销。

## 3. A1 注册

| 项 | 结论 |
|---|---|
| dispatch key | **`AutogradCUDA`**。`probes/thead_reg_probe.py`（每 key 独立进程）：AutogradCUDA 6/7 命中（唯一零命中 `inference_mode`，同 known-issues #19）；纯 `CUDA` 7/7 全命中但反传有"未注册 Autograd kernel"警告 |
| yaml 差异 | **2026-10-25 按 PR36 条目 2 取证收口**: `configs/devices/thead-ppu.yaml` 的 `dispatch_key: CUDA` 是中央 A1 **基键**（有意分层，非笔误），`_profile.py` 的 `AutogradCUDA` 是算子本地注册点——消费点、先例比对与三口径实测见 `PLATFORM.md` §3c-1；复现 `python3 scripts/smoke_dispatch_key.py` |
| autograd | q/k/v 直调反向（flash/efficient 均有 backward op）；可微 additive mask 走 A1 数学 backward |
| 拦截 | op 层 count=3（含 PR36 条目 4 的非方阵 causal 一例）；注册路径与直调路径**逐位一致**（`bitwise_vs_direct=True`）；framework 拦截 28 次 |
| 梯度 | dq/dk/dv vs 原生子进程 max_err 4.88e-4 / 4.88e-4 / 3.91e-3（bf16 容差内）；direct q/k/v + 可微 mask 反向通过（`direct_grad_check=vendor+math-fallback`） |

## 4. 验证结果

```bash
PYTHONPATH=/hosthome/FlagOS-OP python3 example.py thead-ppu          # 三层一键
PYTHONPATH=/hosthome/FlagOS-OP python3 test/kernel_level.py thead-ppu
PYTHONPATH=/hosthome/FlagOS-OP python3 test/kernel_level_tail_thead_ppu.py thead-ppu
PYTHONPATH=/hosthome/FlagOS-OP python3 script/check_accuracy.py --impl triton --device cuda:6
PYTHONPATH=/hosthome/FlagOS-OP python3 probes/guard_check.py thead-ppu
```

| 项 | 结果 |
|---|---|
| kernel 层 | ✅ 37/37，max_err 3.91e-3（bf16），哨兵通过，1k D128 ~0.12ms |
| tail 覆盖（§2d） | ✅ 88/88：S3072 causal ×3 seed、S5000/5120、S5000/5001-bool-mask、zeros/large/boundary + **非方阵 causal 48 案例（8 形状 × 3 dtype × 2 seed）与原生 parity 3 例（PR36 条目 4，见 §6.12）**；boundary 判定用相对容差 5e-3（黄金 extreme 口径），bf16 用 parity_vs_native + 有限子集 rel；带 mask 案例改判「非有限位置与参考逐位一致」（`nonfinite=n/n_ref finite_pos=True`），旧 `n_nonfinite==0` 会把参考自身的合法全遮蔽 NaN 行误判为 FAIL |
| 黄金 | ✅ 397/397，worst 7.812e-3 @ bf16 float-mask（容差 2e-2 内）；extreme fp32 rel 4.988e-3 |
| op 层 | ✅ AutogradCUDA 拦截（count=3）、注册=直调(逐位)、vs 原生 0.0、梯度通过；**非方阵 causal 经 A1 拦截、注册=直调(逐位)、vs 原生 0.0，梯度 vs CPU 参考 dq 1.22e-4 / dk 1.26e-4 / dv 1.92e-3**（PR36 条目 4） |
| 应用层 | ✅ mini-decoder（GQA8/2 causal fp16）拦截 28 次，logits max_diff=0.0，top1/贪心续写一致率 1.00 |
| 守卫 | ✅ 元数据 / cuda:6 PPU 正向路径 / CPU 拒绝 / 注册守卫（`ALL GUARD CHECKS PASS`） |
| perf gate | ✅ `scripts/perf_run.py --device thead-ppu --pattern ops.sdpa --update-baseline` 入库；`perf_compare` FAIL 0 · WARN 0 |
| 仓库一致性 | ◐ `run.py --consistency` FAIL：`ModuleNotFoundError: flag_gems`（A2/gelu_and_mul 锚定门禁，dev 栈结构性缺证，见 §6.11） |

## 5. 性能采样与倒挂收口

数据：[perf_fp16_thead-ppu.json](perf_fp16_thead-ppu.json)、
`perf/baselines/thead-ppu.json`（warmup=20，iters≤100，每次读取一个
输出元素强制完成）。

| shape | ours | Python F.sdpa | FlagGems | ours/F.sdpa |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.092ms | 0.082ms | 缺证 | 0.89x |
| prefill 1k D128 | 0.113ms | 0.104ms | 缺证 | 0.92x |
| prefill 2k D128 | 0.361ms | 0.351ms | 缺证 | 0.97x |
| prefill 4k D128 | 1.136ms | 1.125ms | 缺证 | 0.99x |
| GQA 1k D128 | 0.239ms | 0.205ms | 缺证 | 0.86x |
| decode D128 | 0.053ms | 0.033ms | 缺证 | 0.63x |

仓库 perf gate 台账（B1H16S1024D128 causal fp16）：
`ops.sdpa.thead-ppu.forward` **0.114 ms / 37.70 TFLOPS** vs
`ops.sdpa.native.forward` **0.104 ms / 41.16 TFLOPS → 1.10x 原生**。

**倒挂收口过程**（首版台账 0.426 ms vs 原生 0.105 ms = **4.07x 慢**）：

1. 根因 A：maskless 也走 efficient——其 kernel 本身比 flash/原生慢，
   且输出非连续导致下游拷贝。收口：maskless 分流 flash 直调。
2. 根因 B：`kernel/backends.get_impl()` 每次调用 `find_spec`/
   `importlib` 解析三候选（cProfile 实测 ~80-100 µs/调用）。收口：
   进程级 `_IMPL_CACHE`（栈身份进程内不变）。
3. 两步合计 0.426 → 0.317 → **0.114 ms**，与原生差收敛到 1.10x。

FlagGems 三方对比：dev 容器无 flag_gems（任务书口径"缺一写原因"），
留 thead-ci 第二 profile（P3）补验。

## 6. 关键平台发现

1. **原生 F.sdpa 在 PPU 内部只调 `_scaled_dot_product_flash_attention`**
   （TorchDispatchMode 单行实证）——直调该 overload 即与原生同 kernel。
2. flash schema **无 attn_mask**；带 mask 场景必须走 efficient，
   且 efficient 的 `attn_bias` 有 **16B stride 对齐**硬约束，报错原文
   两种：`(stride(1)*elem)%16!=0` → `attn_bias is not correctly aligned
   (strideH)`；`(stride(2)*elem)%16!=0` → 无后缀
   `attn_bias is not correctly aligned`。修复（列 pad 16B + as_strided）
   **fp32/fp16 已验证等价（9/9）；bf16 2026-10-24 已补验（T4
   `probes/out/thead_bf16_bias_pad.json`）：实测生产 `_aligned_bias` 在 bf16
   非对齐 stride（333/197/1×7/8×333，bool/float mask、causal 0/1 共 11
   组）上 vs 连续 kv_pad 布局直算逐位一致（互差 0）、vs CPU fp32 参考
   bf16 1e-2 abs 黄金口径内全过；两种报错原文在 bf16 均复现。黄金集
   bf16 masked 案例（全 basic S128）走连续 bias 确实不触发非对齐 stride，
   生产 facade 回归 24/24 无回退（worst 7.8125e-3 与交付时历史 worst
   持平）——遗留收口。**
3. efficient schema **无 `enable_gqa`** → GQA 用 `repeat_interleave`。
4. fp32 委托非 extreme 误差 3e-7～5e-6，但 scale=8 饱和 abs 2.79e-3
   超 1e-5 → fp32 走 ATen 组合；**PPU 无 XMLIR bmm JIT 缺陷**
   （S∈(320,640] 设备组合 ≤5.7e-7，无需 p800 的 768 pad）。
5. **bf16 boundary**（V≈dtype_max/2 × 长序列）：厂商 kernel 中间量溢出
   inf，**原生 F.sdpa 同样溢出**——平台分化，判定用 parity_vs_native +
   有限子集 rel<5e-3，不按绝对容差判死。
6. cudnn overload 在 PPU 上 `ACDNN_STATUS_BAD_PARAM`；overrideable 是
   privateuse1 专用——两条"看起来该走"的路都不通。
7. triton 3.6.0 可在 PPU 编译执行（PPUDriver, arch=80），但 `tl.dot`
   精度掉 fp16 量级——fp32 路径不要指望 triton 兜底。
8. dispatch key：`AutogradCUDA` 命中 6/7（`inference_mode` 零命中同
   #19）；纯 `CUDA` 全命中但反传有警告——yaml 里的 `CUDA` 字段与实际
   生效值不同，以 `_profile.py` 检测为准。
9. `get_impl` 的三候选解析必须缓存：`find_spec` 每调用 ~80-100 µs，
   在 0.02ms 级的 kernel 面前就是 3-5x 的差距（本算子 4.07x 倒挂的
   一半根因）。
10. 高精度委托（efficient）不等于高性能：首版"照抄 p800 全走
    efficient"在 PPU 慢 4x——**移植时先证明"与原生同 kernel"再定主路径**。
11. **仓库级 `run.py --consistency` 在本栈结构性缺证**：该门禁锚定
    gelu_and_mul（A2 路由），import `flag_gems`/`vllm_fl`——dev 容器
    均无（`ModuleNotFoundError: flag_gems`），失败与 sdpa/thead 无关。
    sdpa 的"直调↔dispatch 张量级一致"由 `test/op_level.py` 逐位断言覆盖。
12. **非方阵 causal（Sq≠Skv）本可委托，原拒绝是多余的**（PR36 条目 4，
    2026-10-25）：`probes/thead_nonsq_causal_probe.py`（跑前 ppu-smi 6/7
    空闲）实测——原生 `F.sdpa` 对该输入**正确**（aten 语义 = 左上
    `tril(ones(Sq,Skv))`，旧行「与 aten 一致的 Sq==Skv 限制」不成立）；
    **flash 直调 + causal 不生效**（6 形状 × fp16/bf16 err 1.2-1.9，
    所以 maskless 分支不能吃非方阵 causal）；**efficient 直调全对**
    （6 形状 × 3 dtype：≤5.62e-4 / ≤2.98e-3 / ≤2.53e-7，与原生逐案同值）；
    LSE 按 32 倍数 pad（`[..., :Sq]` err ≤9.5e-7）、`_aligned_bias` 与
    GQA 扩 KV 非方阵均等价、厂商反向 dq/dk/dv ≤3.9e-3。故选型 **a**
    （实现按文档路由）：删 `thead_ppu.py` 拒绝行、maskless 非方阵 causal
    改道 efficient、拆 `register.py` 的 `Sq==Skv` 数学 backward 门（旧门
    实测梯度差 **4.68**——非方阵上不折 causal）、`reference.py` 改与
    aten 一致的 top-tril；其余三平台保持显式拒绝（ascend 原为
    `IS_CAUSAL=is_causal and Sq==Skv` **静默降级成不遮蔽**，本批补同款
    拒绝）。判卷黄金集无非方阵 causal 案例，397 例口径不变。
    详见 `PLATFORM.md` §3c-3（四平台矩阵 + 证据表）。

## 7. 复现命令

```bash
cd ops/sdpa
export PYTHONPATH=/hosthome/FlagOS-OP
python3 example.py thead-ppu                                 # 三层一键
python3 test/kernel_level_tail_thead_ppu.py thead-ppu        # §2d 覆盖
python3 script/check_accuracy.py --impl triton --device cuda:6
python3 script/bench_perf.py --device cuda:6 \
    --json-out reports/perf_fp16_thead-ppu.json
python3 probes/guard_check.py thead-ppu
python3 probes/thead_eff_probe.py        # 委托全矩阵（效率/精度/schema）
python3 probes/thead_nonsq_causal_probe.py   # 非方阵 causal 取证（PR36 条目 4，输出 probes/out/）
python3 probes/thead_bias_layout.py      # bias 对齐规则 + kv_pad 规避
python3 probes/thead_bf16_bias_pad.py    # T4 补验: bf16 非对齐 _aligned_bias 等价 + 黄金 masked 子集回归
python3 probes/thead_reg_probe.py AutogradCUDA   # dispatch key 命中
python3 probes/thead_flash_probe.py      # flash/efficient/native 计时
python3 probes/thead_triton_dot_smoke.py # tl.dot 编译+对点（备选记录）
# 仓库 perf 门禁（在仓库根）:
python3 scripts/perf_run.py --device thead-ppu --pattern ops.sdpa --update-baseline
python3 scripts/perf_compare.py --device thead-ppu
```

## 8. 结论与遗留

可合入。遗留（不阻断本次交付）：

- ~~bf16 非对齐 stride 的 `kv_pad` 未单独验证（§6.2）~~
  **2026-10-24 已补验收口（T4）**：`kv_pad` 即 efficient 委托路径的
  attn_bias 16B 对齐修复 `_aligned_bias`（T1 b29a1ac message 原话
  "bf16 非对齐 kv_pad 未单独验证"）。bf16 补验+黄金 masked 子集回归见
  §6#2 与 `probes/out/thead_bf16_bias_pad.json`；黄金 bf16 masked 形状口径
  （basic S128 连续 bias）不触发该分支的论证也已落证;
- `inference_mode` 下 AutogradCUDA 零命中（known-issues #19 同款，
  推理方需 `no_grad` 或纯 `CUDA` key 注册方案）；
- FlagGems 三方性能对比缺证（P3，thead-ci 第二 profile）；
- ~~`configs/devices/thead-ppu.yaml` 的 `dispatch_key: CUDA` 与实测生效值
  `AutogradCUDA` 不一致——configs 改动需先上报（任务书 §3），
  当前由 `_profile.py` 检测覆盖。~~
  **2026-10-25 已按 PR36 条目 2 取证收口**：保留 YAML 基键 `CUDA`（中央 A1
  消费点读取，grad/no_grad/inference_mode 三口径实测各拦 1 次）与算子本地
  `AutogradCUDA` 分层——先例比对（p800-kunlunxin{,-gems5}=CUDA、
  cambricon=PrivateUse1 同型）与取证路径写入 `PLATFORM.md` §3c-1 及
  thead-ppu.yaml 注释；新增 `scripts/smoke_dispatch_key.py` 三断言
  （注册 key / 反向路径 / no_grad·inference 一致）跑绿。
