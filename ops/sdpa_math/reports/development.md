# 算子开发报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

| 项 | 值 |
|---|---|
| 算子名称 | `aten::_scaled_dot_product_attention_math`（SDPA math 后端，双输出） |
| 实现路线 / 开发级别 | **A1** / Triton 级（NPU）+ P800 fast（vendor O/LSE + exact-P）+ torch 级（CPU/fallback） |
| 目标设备 | `ascend910` · `p800-kunlunxin` · `cpu`（参考与梯度兜底） |
| 日期 / 状态 | 2026-09-30~10-01 / 定稿 |

## 1. 环境配置

| 项 | 值 |
|---|---|
| 环境 | Ascend910 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0（`torch.npu.is_available()=True`） |
| triton | 3.5.1（ascend 后端） |
| flag_gems | 5.3.5 |
| python | 3.11.15 |
| 设备 | Ascend910_9382（`npu:0`），`configs/devices/ascend910.yaml` |
| A1 注册点 | `AutogradPrivateUse1`（+ 成对 `PrivateUse1`），见 §3 |
| 缺失 | `torch_mlu` 未安装；`configs/env/ascend910.lock.yaml`
不存在 → `scripts/check_env.py --device ascend910` 不可用，环境以本表为准 |
| 消费方限制 | `F.sdpa` 在 NPU 上被 torch_npu 路由到融合注意力，**命中本算子 0 次**（[probes/native_semantics.py](../probes/native_semantics.py) §5）→ NPU 消费方直调 `torch.ops` |

### 1.1 P800 环境（2026-10-01 复验）

| 项 | 值 |
|---|---|
| torch / python | `2.9.0+cu129` / Python 3.10 |
| torch_xmlir | `XPYTORCH XMLIR--bc1b1dc6f-dev+2026032411` · XTDK 3.6.0.1 |
| triton / FlagGems | `3.0.0+03c4c9be` / `4.2.1rc0` |
| 设备 / A1 key | `cuda:1`（锁定镜像稳定卡）· `AutogradCUDA+CUDA` 成对注册 |
| 消费方 | FlagOS/FlagGems 栈内，`flag_gems.only_enable(['gelu'])` 启动周边 GELU |

## 2. 算子定义

- **签名**（torch 2.10 实测）:
  `aten::_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)`
- **语义**: [reference.py](../reference.py)。第二输出是注意力概率图
  `attn_probs (B,Hq,Sq,Skv)`——本算子与 `F.sdpa` 的本质差别（后者丢弃 P）。
  内部 fp32（fp64 输入保持 fp64，仅用于 gradcheck），返回 dtype=query.dtype。
- **容差**（[docs/acceptance.md](../../../docs/acceptance.md) 分档）:
  fp32 `1e-5` abs · fp16/bf16 `2e-2` abs · 极端用例（`scale=0.5` + 全遮蔽行/
  病态大行）`1e-3` rel。
- **shape 清单**（黄金 175 组 + kernel 52 组，均含非整倍数维度）:
  `S ∈ {1,4,6,7,100,128,197,256,48}` 等，`D ∈ {16,64,80,128}`，
  `Hq/Hkv ∈ {4/4, 8/2, 16/16, 32/8}`，`Sq≠Skv`（6×4、4×6、1×7），
  掩码 `bool4d / float4d / bool2d / bool_b1`，`dropout_p ∈ {0,0.2,0.3,0.5}`。

## 3. 实现说明

**路线选择 A1**（[集成指南](../../../docs/integration.md)）: 本算子是
`CompositeImplicitAutograd`，任意 backend key 的 Python impl 都能覆盖它
（[probes/native_semantics.py](../probes/native_semantics.py) §1 的
dispatch dump），旧代码零改动。

四个实现（同一语义，独立写法 / 平台专属加速）:

| 级别 | 文件 | 作用 | 要点 |
|---|---|---|---|
| reference | `reference.py` | 判卷标准 | `masked_fill` + exp/sum 守卫；`_validate` 集中校验（causal+mask 冲突 / GQA 整除 / K-V 头一致） |
| torch | `kernel/torch_level.py` | CPU 交付 + 第二判卷人 | `where` 生成 causal bias、`nan_to_num(softmax)` 守卫（与 reference 不同写法） |
| triton | `kernel/triton_level.py` | NPU 交付 | 两段式：`_probs_kernel`(pass1 行最大 + pass2 归一) → `_pv_kernel`；`_score_block` 两 pass 共用；固定 64×64 tile；dropout 用 ATen 收口。**小形状按 score tile 数（≤56）自动改走单 kernel 融合**（QK→softmax→PV 一趟，见 [performance.md §6/§9](performance.md)） |
| p800 | `kernel/p800_fast_level.py` | P800 交付 | vendor efficient attention 产出 `O/LSE`；`P=exp(scale·QKᵀ-LSE)` 保留双输出契约；mask/dropout/fp32/direct-autograd 回退 torch |

**Triton 关键决策**:
1. 必须物化 P（算子契约），故不能用 flash 的重算策略 → 两段式而非单遍；
   小形状例外：tile 数 ≤56 时单 kernel 融合（P 块只在寄存器里、O 直写）
   省一次 launch，阈值与实测见 [performance.md §6/§9](performance.md)；
2. `flag_gems.runtime.torch_device_fn.device` 上下文包裹启动（#11）；
3. 尾块 masked load（#15a），不用 `@triton.autotune`（#15b）；
4. fp32 走 `input_precision="ieee"`；
5. dropout 不进设备码：随机路径 `a=keep/(1-p)` 一次性乘、显式路径分别
   产出 `probs`（返回用）与 `probs_pv`（PV 用），PV 启动传
   `probs_pv.stride(...)`——因为 native 两条 dropout 规则下 `out ≠ P@V`。

**注册（`register.py`）**:
- 成对注册 `[_PAIRS]`（`AutogradPrivateUse1↔PrivateUse1`、
  `AutogradCPU↔CPU`）——单键的两个实测反例见 §6；
- `impl_fn_` 补 schema 默认值、计数、`not torch.is_grad_enabled()` 时
  绕开 `autograd.Function`（inference 张量不能 `save_for_backward`）；
- forward 以 `dropout_p=0` 调 impl，dropout 在 `_fwd` 统一落地（随机 keep
  只生成一次 → 反向确定）；backward 用预 dropout 的 P0 做 softmax 雅可比，
  两路梯度 `dO`/`dprobs` 在 `dP0` 处汇合，fp64 输入按 fp64 累积；
- 每次注册持一份 `_impl`（子类绑定），同进程可并存 npu=triton +
  CPU=torch 或 P800=p800 + CPU=torch（gradcheck/F.sdpa 消费方）。

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ☑ PASS | 52 组（16 shape × 3 dtype + 掩码/边界/错误路径）×3 profile；最差（NPU）fp32/fp16/bf16 见 reports/accuracy.md §2；哨兵确定性+敏感；dropout 显式一致 + 随机 keep=0.499、×2.000 |
| 框架层 | ☑ PASS | 21 项：四模式拦截（grad/no_grad/inference_mode/无 requires_grad）、注册=直调逐位、vs 原生 3.6e-7、6 组 fp64 gradcheck、`F.sdpa(MATH)` 拦截 + 输出=参考；拦截计数 3822 |
| 应用层 | ☑ PASS | 8 项：mini-decoder + 概率图消费者，拦截 21 次，logits L∞ 1.86e-8，top1=1.000，贪心序列 1.000，全权重梯度 L∞ 1.75e-10 |
| 黄金回归 | ☑ 175/175 | reference / torch(CPU) / triton(NPU) / p800(P800) 四实现全过；`--impl native` 131/131（44 组 bool 按 §2 分歧跳过） |
| perf 门禁 | ☑ FAIL 0 | `perf_run --pattern sdpa_math` + `perf_compare`：triton 0.543ms(+7.8%) / torch 0.567ms / reference 0.874ms（warmup 20 · iters 50，融合改动后复测） |
| P800 | ☑ PASS | kernel 52/52；黄金 p800 @ cuda:1 175/175（worst 3.906e-3，bf16 容差内）；A1 21/21；FlagGems 应用层 9/9；P800 perf gate FAIL 0 |
| FlagOS E2E | ☑ PASS | 4 层 mini-LLM + GELU/GQA/熵正则/训练/生成；A1 拦截 856 次，logits L∞ 1.56e-2，贪心序列一致 |

## 5. 性能

NPU fp16（`script/bench_perf.py`，speedup = 原生 math / ours，>1 更快）:

| shape | ours | 原生 math | speedup | A1 路径 |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.515ms | 0.626ms | **1.22x** | 0.555ms |
| prefill 1k D128 | 0.538ms | 0.695ms | **1.29x** | 0.572ms |
| prefill 2k D128 | 1.369ms | 2.776ms | **2.03x** | 1.402ms |
| GQA 1k D128 | 0.810ms | 1.351ms | **1.67x** | 0.863ms |
| decode 64 D128 | 0.217ms | 0.255ms | **1.17x** | 0.295ms |
| tail100 D64 | 0.219ms | 0.253ms | **1.16x** | 0.295ms |

CPU fp32 同口径 1.05-1.40x（6 形状全过）。**精度代价**: 零——两实现
同走黄金 175/175（自研 worst 3.9e-3 / 原生 1.95e-3，同为 bf16 量化级）。
详细分析见 [reports/performance.md](performance.md)。

### P800 / Kunlunxin

P800 生产实现为 `kernel/p800_fast_level.py`，A1 成对注册到
`AutogradCUDA+CUDA`。实现路径：

```text
aten::_scaled_dot_product_efficient_attention(query,key,value)
    ↓ 返回 output + row-wise LSE
P = exp(scale · QKᵀ - LSE)   # 半精度 QK，免行 max/sum 归约
causal 上三角 / 尾块          # 显式置 0
    ↓
return (output, P)            # 仍是 exact sdpa_math 契约
```

因本算子必须返回概率图，不能只返回 efficient attention 的 `output`。
P800 fp16 direct **1.46-2.21x** native，A1 **1.31-2.17x**；原始数据见
`reports/perf_p800-kunlunxin.json`。

`script/bench_f_context.py` 额外记录 no-P `F.sdpa` 参考列：原
`torch_level` exact-contract math 在 P800 fp16 上慢 **2.45-8.51x**。
这不是公平 A/B，因为 `F.sdpa`
只返回 `out`，可不物化 `(B,Hq,Sq,Skv)` 概率图。它提示 P800 仍有内部融合
空间，但目标应是减少 `scores→softmax→P→PV` 的中间读写，而不是达到
no-P FlashAttention。

启用 fast path 后，该参考差距收敛为 **1.98-4.46x**。剩余成本来自额外
QK、GQA `K` expansion、causal mask 构造与全量 `P` 写出。

`script/e2e_flagos.py` 将该路径放进 4 层 mini-LLM 端到端验证：FlagGems
GELU 负责周边 MLP，A1 负责目标 private math op，`P` 被熵正则消费。推理与
生成链 **1.02-1.12x**；CE+entropy 训练链 **0.89-1.00x**，说明自定义
backward 是训练场景的下一个瓶颈。详见
[reports/e2e_flagos_p800-kunlunxin.json](e2e_flagos_p800-kunlunxin.json)。

## 6. 已知问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | `bool attn_mask` **直调**是 0/1 加性怪癖（`probes` §3 实测 0/0 vs `-inf` 遮蔽 0.514） | 只影响绕过 `F.sdpa` 直调 bool mask 的调用方；本实现按 `-inf` 遮蔽（对齐 `F.sdpa`） | 有意分歧，黄金 44 组在 `--impl native` 下跳过并注明 |
| 2 | 只注册 `Autograd*` → `torch.inference_mode()` 不命中（counter 不增）；反向报 "an autograd kernel was not registered" | 推理/守卫路径绕过自研实现 | **已修**：成对注册（`probes` §6 前后对照） |
| 3 | 朴素 Python impl 不建图（Triton launch 不产生 grad_fn） | 反向断裂 | **已修**：`autograd.Function` + 数学 backward |
| 4 | NPU decode/tail 直调经单 kernel 融合后 1.16-1.17x；**A1 包装路径仍 0.86x** | 包装 ~0.078ms 对 0.29ms 级调用占比高，且在 autograd 图节点层 | 接受（[performance.md §6](performance.md)）；P800 fast 的 A1 小形状为 1.31-1.53x |
| 5 | `dropout_p>0` 的 A1 路径 `out` 走 ATen matmul 而非 fused PV | 仅 dropout 场景，非性能路径 | 接受（`register.py` 已注明） |
| 6 | `torch.library` 无官方撤销 API（`unhook_a1` 仅删引用） | 同进程注册不可回滚 | 接受：基线须在注册前采集（bench/test 已按此实现） |
| 7 | P800 源码级 exact-P Triton 双 pass 会触发 XMLIR pointer-state rewrite 失败；已有 no-P custom schedule 也慢于 vendor efficient | 不能直接复用 Ascend Triton kernel | 生产改走 vendor `O/LSE` + exact-P 组合；后续需等/换 XMLIR lowering |
| 8 | FlagOS E2E 训练链 0.89-1.00x（推理/生成 1.02-1.12x） | A1 自定义数学 backward 与 `dprobs` 熵正则通路抵消部分 forward 收益 | 记录为训练场景后续优化项 |

## 7. 结论与后续

对照 [Must 清单](../../../docs/acceptance.md#levels)：必测矩阵 ☑（非整倍数
shape + 三 dtype + 特殊用例）· 哨兵 ☑ · 黄金 100% ☑ · 三层全绿 ☑ ·
性能门禁 FAIL 0 ☑ · 报告四件套 ☑ → **可交付**。

后续优化方向（非阻塞）:
1. ~~decode 小形状：合并 pass1/pass2、减少 launch 次数~~ **已完成**
   （单 kernel 融合：decode 0.287→0.217ms、tail 0.285→0.219ms，
   见 [performance.md §6](performance.md)）；
2. 4k+ 长序列：P 分块落盘 + 分段 PV，压峰值显存；
3. `scale` 为 tensor 的慢路径（当前转 python float）；
4. P800 更进一步：若 XMLIR 支持稳定 exact-P 写出调度，将
   `QK→exp(·-LSE)→P store` 合并为单 kernel，减少 ATen 中间写。
5. 训练链 backward：针对 CE+`dprobs` 消费方优化数学反传，减少 fp32
   中间张量与 Python autograd 开销。

## 附录: 复现命令

```bash
cd ops/sdpa_math
python3 probes/native_semantics.py ascend910     # §1/§3/§6 证据
python3 example.py ascend910                     # 三层一键
python3 script/gen_golden.py && \
python3 script/check_accuracy.py --impl triton --device npu:0 && \
python3 script/check_accuracy.py --impl native --device npu:0
python3 script/bench_perf.py --device npu:0 --register \
  --json-out reports/perf_ascend910.json
python3 example.py p800-kunlunxin
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
python3 script/bench_f_context.py --device cuda:1 \
  --json-out reports/perf_f_context_p800-kunlunxin.json
python3 script/e2e_flagos.py --device cuda:1 --dtype bfloat16
cd ../.. && python3 scripts/perf_run.py --device ascend910 --pattern sdpa_math
python3 scripts/perf_compare.py --device ascend910
```
