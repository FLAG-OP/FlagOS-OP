# sdpa_math-op: `aten::_scaled_dot_product_attention_math`（SDPA math 后端 + 概率图）

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
复用同一语义参考、黄金数据、三层测试与 A1 注册链。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::_scaled_dot_product_attention_math`（torch SDPA 的 **math 后端**，`CompositeImplicitAutograd`） |
| 语义 | `softmax(QKᵀ·scale + mask)` 双输出：`(out, attn_probs)` · GQA · causal · bool/float mask · dropout 双规则 · fp32 内部（fp64 输入保持 fp64，供 gradcheck） |
| 路线 | **A1** aten 拦截（成对注册 `Autograd*` + 纯设备键），旧业务代码零改动 |
| 平台 | **ascend910**: 自研 Triton；**p800-kunlunxin**: ATen 组合 + A1；**cpu**: ATen 组合（对照/梯度兜底） |
| 验证 | kernel 52 组 ×3 profile · 黄金 **175/175** · 原生对照 **131/131**（44 组 bool 跳过）· op 21 项（6 gradcheck）· 应用层 8-9 项 ✅ |
| NPU 性能（fp16，vs 同为"返回 out+P"的原生 math） | prefill1k D64 **1.20x** · 1k D128 **1.32x** · 2k D128 **2.05x** · GQA **1.74x** · decode 0.92x |
| CPU 性能（fp32，同口径） | 1.05-1.40x（6 形状全过） |
| P800 性能（fp16，同口径） | direct **1.03-1.27x** native；A1 大 shape **1.02-1.24x** |

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
python3 script/check_accuracy.py --impl torch    --device cuda:1
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl native   --device npu:0   # 原生对照

python3 script/bench_perf.py --device npu:0 --register \
  --json-out reports/perf_ascend910.json
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
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
├── reference.py              # CPU 语义参考（判卷标准，含 dropout 双规则顶注）
├── register.py               # A1 成对注册 + autograd.Function（dout/dprobs 两路）
├── _profile.py               # ascend910 / p800-kunlunxin / cpu 本地 profile
├── kernel/
│   ├── triton_level.py       # 两段式 _probs_kernel + _pv_kernel（NPU）
│   └── torch_level.py        # ATen 组合（独立写法，非 reference 转发）
├── test/                     # kernel / op / framework 三层
├── goldendata/               # 声明式 175 组黄金（inputs_spec.yaml）
├── script/                   # gen_golden / check_accuracy / bench_perf
├── probes/native_semantics.py# native 语义证据（schema/参数形态/dropout 表/注册键）
└── reports/                  # development / test-report / accuracy / performance
```

## P800 / Kunlunxin 路线

P800 生产实现选择 `torch_level.sdpa_math_torch`：

```text
A1 AutogradCUDA + CUDA 成对注册
    ↓
torch_level ATen composition
```

本算子必须 materialize 第二输出 `P`，不能直接替换为 efficient attention。
当前 torch 组合与原生 private math 同为 ATen 路径，P800 实测 direct
**1.03-1.27x** native，A1 大 shape **1.02-1.24x**。

框架层运行在 FlagOS 算子栈内：

```python
import flag_gems
flag_gems.only_enable(include=["gelu"])
```

锁定镜像上全量 `flag_gems.enable()` 非确定，因此选择稳定且真实被消费方
调用的 GELU 作为 surrounding op；目标 `sdpa_math` 走 A1。

## 硬约束（本栈已内建）

1. Triton 启动必须包 `flag_gems.runtime.torch_device_fn.device` 上下文（#11）
2. N 尾块归约安全（pad / 两阶段）（#15a）
3. 不用 `@triton.autotune`（#15b 会选出非法 `num_warps`）
4. `register_a1` 返回的 lib **必须持有引用**（否则注册被 GC 掉）
