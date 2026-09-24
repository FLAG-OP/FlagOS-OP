# sdpa-op: `aten::scaled_dot_product_attention` 多平台实现

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
复用同一语义参考、黄金数据、三层测试与 A1 注册链。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::scaled_dot_product_attention`（SDPA / flash-attention 族） |
| 语义 | `softmax(QKᵀ·scale + mask)·V` · GQA · causal · bool/float mask · fp32 内部 |
| 路线 | **A1** aten 拦截，旧业务代码零改动 |
| 平台 | **ascend910**: 自研 Triton online-softmax；**p800-kunlunxin**: 厂商 efficient-attention 委托 + fp32 精度补偿；**mlu590**: TMO FA + fused overrideable（CNNL）委托 |
| 公共入口 | `kernel/triton_level.py` facade → `kernel/backends/{ascend910,p800_kunlunxin,mlu590}.py` |
| 验证 | P800: kernel 37/37 · 黄金 397/397 · A1 拦截/梯度 · mini-decoder ✅；MLU590: kernel 37/37 · 黄金 397/397 · 三层/守卫 ✅；Ascend 历史 265/265 保留 |
| P800 性能 | 强制读回输出的 fp16 采样相对 Python `F.sdpa` 加速 **1.02-1.14x**；FlagGems 2k/4k 比 ours 慢 5.1-6.6x |
| MLU 性能 | TMO+fused 相对原生 **1.00-1.33x**；FlagGems 慢 10-40x（仅兜底/对照） |

平台绑定与坑位见 [PLATFORM.md](PLATFORM.md)，多平台扩展流程见
[MERGE.md](MERGE.md)。

## 快速开始

```bash
# 本机有 torch_mlu 时自动选 mlu590；有 torch_xmlir 选 p800；Ascend 可显式传 ascend910
python3 example.py mlu590
python3 example.py p800-kunlunxin

python3 test/kernel_level.py mlu590
python3 test/op_level.py mlu590
python3 test/framework_level.py mlu590
python3 probes/guard_check.py mlu590

# CPU 黄金生成后，同一 397 组可直接测任意平台
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl triton --device mlu:0
python3 script/check_accuracy.py --impl triton --device cuda:1

python3 script/bench_perf.py --device mlu:0 \
  --json-out reports/perf_fp16_mlu590.json
python3 script/bench_perf.py --device cuda:1 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 script/bench_cross_platform.py --device mlu:0
python3 script/bench_cross_platform.py --device cuda:1
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.sdpa
python3 scripts/perf_compare.py --device p800-kunlunxin
```

设备映射遵循 `configs/devices/mlu590.yaml`（`mlu:0`）与
`configs/devices/p800-kunlunxin.yaml`（`cuda:1`）。

## 目录要点

```
sdpa/
├── reference.py              # CPU fp32 语义参考
├── register.py               # A1 注册 + autograd.Function
├── _profile.py               # ascend910 / p800-kunlunxin / mlu590 本地 profile
├── kernel/
│   ├── triton_level.py       # 兼容 facade（旧 import 不变）
│   ├── backends/ascend910.py
│   ├── backends/p800_kunlunxin.py
│   ├── backends/mlu590.py
│   ├── torch_level.py        # ATen 组合对照
│   └── _native_shim.py
├── test/                     # kernel / op / framework 三层
├── goldendata/               # 声明式 397 组黄金
├── script/                   # 精度、性能、跨平台基准
├── probes/                   # 注册/守卫/平台探针
└── reports/
```

## P800 实现策略

1. **优先复用厂商成熟 kernel**: fp16/bf16 直调
   `aten::_scaled_dot_product_efficient_attention`，bool mask 转为
   additive `-inf` bias，并恢复全遮蔽行为 NaN 语义。
2. **fp32 不直接委托**: 厂商 kernel 对 fp32 的误差约 `2e-4`，超过
   CPU 黄金 `1e-5`；fp32 走 ATen 组合。
3. **绕开 XMLIR bmm 缺陷**: 实测 `S∈(320,640]` 可触发 `bmm_one_loop`
   JIT 编译失败；该区间 pad 到 768（更长边保持原长）后用合法 mask
   还原结果。
4. **A1 key 用 `AutogradCUDA`**: dispatcher 可能省略 schema 默认值，
   注册 wrapper 先补齐参数再进入统一数学 backward。
5. **直接调用保留 autograd**: q/k/v 反传时前向补算 efficient kernel 的
   log-sumexp；可微 float mask 走厂商 backward 会报
   `bias_requires_grad not supported yet`，自动复用 A1 数学 backward。

### 自研调度实验

`kernel/p800_custom_triton.py` 保留了一个未接入生产 backend 的固定调度
Triton forward。它通过 no-mask causal、GQA、非 causal 和尾块精度检查，
但 bool/float mask 仍 launch 失败，且比厂商 efficient attention 慢
**2.1-6.5x**。实验结论见
[reports/p800-custom-schedule.md](reports/p800-custom-schedule.md)。

## mlu590 实现策略（Cambricon）

1. **P0 主路径 = fused overrideable**: 直调
   `aten::_scaled_dot_product_fused_attention_overrideable`（CNNL FA v2，
   与原生 `F.sdpa` 同路径）≈1.0x 原生；**不可**在 backend 内再调
   `F.sdpa`（A1 注册后递归）。
2. **P1 快路径 = TMO FA**: `torch_mlu_ops.flash_attention`
   （`cnnlScaledDotProductAttn_v7`，BSHD 布局，bias BHSD）；半精度
   prefill 实测 **1.21-1.33x** 原生；`SDPA_MLU_TMO=0` 强制回 P0。
3. **bool mask 转 additive `-inf` bias**；**causal+mask 先折叠进 mask**；
   **全遮蔽行恢复 NaN**。
4. **GQA**: overrideable 无 `enable_gqa` → `repeat_interleave` 扩 KV。
5. **兜底**: math private → FlagGems `_cambricon` → fp32 reference。
6. **A1 key = `AutogradPrivateUse1`**: 与 torch_npu 同栈结论；注册守卫
   放宽为 torch_npu **或** torch_mlu 可用；可微直调复用 A1 数学 backward。

复现与性能数字见 [reports/mlu590.md](reports/mlu590.md)。

## 应用层

P800 框架/应用验证运行在 FlagOS 算子栈内：

```python
import flag_gems
flag_gems.only_enable(include=["gelu"])  # mini-decoder 的 surrounding op
```

锁定镜像上全量 `flag_gems.enable()` 在该 consumer 上存在非确定性，因此
选择稳定且真实被模型调用的 GELU 作为 FlagOS/FlagGems 代表路径；
SDPA 由本目录 A1 注册接管。

`test/framework_level.py` 构建 Llama 风格 4 层 mini-decoder
（GQA 8/2、causal），业务代码只调用
`F.scaled_dot_product_attention`。P800 使用 bf16 规避该随机模型
Linear/LayerNorm 的 fp16 溢出；注册后拦截 28 次，logits 与基线
逐位一致，贪心续写一致率 1.00。

## 关键平台发现

1. XMLIR 将 Kunlun XPU 暴露为 CUDA tensor；仅凭 `device.type=="cuda"`
   会误接 NVIDIA，P800 backend 额外要求 `torch_xmlir` 可导入。
2. `AutogradCUDA` 注册可稳定拦截 Python `F.sdpa`，但必须补齐默认参数。
3. Python `F.sdpa` 与底层 efficient kernel 的选择/开销不同；直调 private
   aten efficient 入口在典型 fp16 shape 上更快。
4. FlagGems `_kunlunxin` attention 无 mask/causal 路径可运行但慢；
   float mask 场景实测 `xpuLaunchKernel ... Operation not permitted`。
   固定 tile 直测在 FA2 协议下比厂商 efficient attention 慢
   5.4-7.0x；raw-pointer 改写实验见
   [reports/p800-triton-probe.md](reports/p800-triton-probe.md)。

## Ascend 910 保留能力

Ascend 版仍是完整 Triton 主实现（online-softmax two-pass、causal 截断、
GQA/双 mask/尾块），历史精度、性能与根因分析见
[REPORT.md](REPORT.md)、[reports/development.md](reports/development.md)、
[reports/performance.md](reports/performance.md)。

## 关键平台发现（MLU）

1. **原生 `F.sdpa` 走 fused overrideable（CNNL FA v2），不是 math**；
   直调 math private 会拿到未融合分解（0.11-0.55x 原生）。
2. MLU 上 efficient/flash/cudnn 三个 aten private op 仍 CPU fallback
   失败；可用融合入口是 **fused overrideable** 与 **TMO flash_attention**。
3. TMO 为 BSHD 布局、bias 形状 `(B,H,Sq,Skv)`；半精度 prefill 再快
   1.2-1.8x（见 reports/mlu590.md）。
4. math private 直接收 bool 在本栈 err≈0.19，必须转 additive。
5. FlagGems `_cambricon` attention 精度过黄金但慢 10-40x 且无 autograd。
6. 全遮蔽行 fused/TMO 返回有限值，需显式恢复 NaN（同 p800 教训）。
7. `SDPA_MLU_TMO=0` 关闭 TMO，只走 overrideable。
