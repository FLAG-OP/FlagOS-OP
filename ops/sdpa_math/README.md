# sdpa_math-op: `aten::_scaled_dot_product_attention_math`（SDPA math 后端 + 概率图）

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
复用同一语义参考、黄金数据、三层测试与 A1 注册链。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::_scaled_dot_product_attention_math`（torch SDPA 的 **math 后端**，`CompositeImplicitAutograd`） |
| 语义 | `softmax(QKᵀ·scale + mask)` 双输出：`(out, attn_probs)` · GQA · causal · bool/float mask · dropout 双规则 · fp32 内部（fp64 保持；P800 半精度 fast path 见下） |
| 路线 | **A1** aten 拦截（成对注册 `Autograd*` + 纯设备键），旧业务代码零改动 |
| 平台 | **ascend910**: 自研 Triton（两段式 probs + PV，小形状单 kernel 融合）；**p800-kunlunxin**: vendor O/LSE + exact-P + A1；**cpu**: ATen 组合（对照/梯度兜底） |
| 验证 | kernel 52 组 ×3 profile · 黄金 **175/175** · 原生对照 **131/131**（44 组 bool 跳过）· op 22 项（6 gradcheck + fused backward）· 应用层 8-9 项 · FlagOS E2E ✅ |
| NPU 性能（fp16，vs 同为"返回 out+P"的原生 math） | prefill1k D64 **1.22x** · 1k D128 **1.29x** · 2k D128 **2.03x** · GQA **1.67x** · decode **1.17x** · tail100 **1.16x** |
| CPU 性能（fp32，同口径） | 1.05-1.40x（6 形状全过） |
| P800 性能（fp16，同口径） | direct **1.46-2.21x** native；A1 **1.31-2.17x** |

## 快速开始

```bash
python3 example.py ascend910          # 三层一键（kernel/op/framework）
python3 example.py cpu
python3 example.py p800-kunlunxin

python3 test/kernel_level.py ascend910
python3 test/op_level.py ascend910
python3 test/framework_level.py ascend910

# 黄金（CPU 生成一次，跨平台复用）
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl triton   --device npu:0
python3 script/check_accuracy.py --impl p800     --device cuda:1
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl native   --device npu:0   # 原生对照

python3 script/bench_perf.py --device npu:0 --register \
  --json-out reports/perf_ascend910.json
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
python3 script/bench_f_context.py --device cuda:1 \
  --json-out reports/perf_f_context_p800-kunlunxin.json  # no-P 参考列
python3 script/e2e_flagos.py --device cuda:1 --dtype bfloat16
python3 probes/native_semantics.py ascend910     # native 语义证据表
```

## 语义要点（`reference.py` 顶注 + `probes/native_semantics.py` 实测）

1. **双输出**: `(out, attn_probs)`，`attn_probs = (B, Hq, Sq, Skv)` 是
   注意力概率图——蒸馏/熵正则类消费方会用它，backward 必须支持
   `dprobs` 通路（`register.py` 的 autograd.Function 两路梯度）。
2. **dropout 两条 native 规则**（最容易抄错）:
   - **显式 `dropout_mask`**: `keep=(mask!=0)`（幅值被忽略），`p=0` 时
     掩码**完全被忽略**；返回 `P = softmax⊙keep`（**不带** `1/(1-p)`），
     `out = (softmax⊙keep/(1-p))@V` → **`out ≠ P@V`**；
   - **随机路径**（`F.sdpa` 唯一会走的形态）: `P = softmax⊙keep/(1-p)`、
     `out = P@V`（自洽）。
   两条路径在 `register.py` 统一落地（impl 一律以 `dropout_p=0` 调用），
   反向复用同一张 keep → 完全确定。
3. **`F.sdpa` 到达本算子的形态**: bool mask 已被 dispatcher 转成
   float32 `-inf` 加性、`dropout_mask` 恒为 `None`、`scale/enable_gqa`
   走 kwargs；**NPU 上 `F.sdpa` 被 torch_npu 路由到融合注意力、命中数
   = 0** → NPU 消费方只能直调 `torch.ops`（`framework_level.py` 即此）。
4. **唯一有意分歧**: `bool attn_mask` **直调**按 0/1 加性怪癖，与
   `F.sdpa` 的 `-inf` 遮蔽不一致；本实现按遮蔽（对齐 `F.sdpa`），
   `check_accuracy --impl native` 对 44 组 bool 用例跳过并注明。
5. **冲突与边界**: `causal + attn_mask`（bool/float 皆然）报
   `Explicit attn_mask should not be set...`；全 `-inf` 行 → `P=0、out=0`
   （不是 NaN）；`enable_gqa` 要求 `Hq % Hkv == 0`。

## 目录要点

```
sdpa_math/
├── __init__.py              # 包化入口：from ops.sdpa_math import register_a1
├── reference.py              # CPU 语义参考（判卷标准，含 dropout 双规则顶注）
├── register.py               # A1 成对注册 + autograd.Function（dout/dprobs 两路）
├── _profile.py               # ascend910 / p800-kunlunxin / cpu 本地 profile
├── kernel/
│   ├── triton_level.py       # 两段式 _probs_kernel + _pv_kernel（NPU）
│   ├── p800_fast_level.py    # vendor O/LSE + exact-P（P800 fast path）
│   └── torch_level.py        # ATen 组合（独立写法，非 reference 转发）
├── test/                     # kernel / op / framework 三层
├── goldendata/               # 声明式 175 组黄金（inputs_spec.yaml）
├── script/                   # gen_golden / accuracy / perf / F-context / FlagOS E2E
├── probes/native_semantics.py# native 语义证据（schema/参数形态/dropout 表/注册键）
└── reports/                  # development / test-report / accuracy / performance
```

## P800 / Kunlunxin 路线

P800 生产实现选择 `p800_fast_level.sdpa_math_p800_fast`：

```text
A1 AutogradCUDA + CUDA 成对注册
    ↓
aten::_scaled_dot_product_efficient_attention → output + LSE
QKᵀ → exp(scale·QKᵀ - LSE) → full P
return (output, P)
```

本算子必须 materialize 第二输出 `P`，不能直接替换为 efficient attention。
fast path 只在 fp16/bf16、无 mask、无 dropout 时启用；fp32、mask、dropout
与 direct-autograd 场景回退 `torch_level`。P800 实测 direct
**1.46-2.21x** native，A1 **1.31-2.17x**。

`script/bench_f_context.py` 另提供 no-P `F.sdpa` 参考列：P800 fp16 下
exact-contract math 约慢 **1.98-4.46x**。这不是同输出结论，因为 `F.sdpa`
不返回 `P`、也不物化概率图；它说明后续 P800 内部融合有机会继续压缩
QK/GQA expansion/mask/P 写出，但受“必须写全量 P”限制，不应以
no-P FlashAttention 作为可达目标。源码级 exact-P Triton 双 pass 目前会触发
XMLIR pointer-state rewrite 限制，故先采用 vendor LSE 组合。

框架层运行在 FlagOS 算子栈内：

```python
import flag_gems
flag_gems.only_enable(include=["gelu"])
```

锁定镜像上全量 `flag_gems.enable()` 非确定，因此选择稳定且真实被消费方
调用的 GELU 作为 surrounding op；目标 `sdpa_math` 走 A1。

端到端入口 `script/e2e_flagos.py` 自建 4 层 Llama 风格 mini-LLM
（D128 · GQA 8/2 · vocab1024），每层直调本算子并把 `P` 送入注意力熵
正则，覆盖 forward、backward、8 步贪心生成与 FlagGems GELU。实测
推理 forward **1.07x**、仅 CE 训练 **1.45x**、CE+概率图熵正则训练
**1.48x**，多步生成 **1.09x**；fused backward off/on A/B **1.45-1.50x**。
原始数据见
[reports/e2e_flagos_p800-kunlunxin.json](reports/e2e_flagos_p800-kunlunxin.json)。

## 硬约束（本栈已内建）

1. Triton 启动必须包 `flag_gems.runtime.torch_device_fn.device` 上下文（#11）
2. N 尾块归约安全（pad / 两阶段）（#15a）
3. 不用 `@triton.autotune`（#15b 会选出非法 `num_warps`）
4. `register_a1` 返回的 lib **必须持有引用**（否则注册被 GC 掉）
