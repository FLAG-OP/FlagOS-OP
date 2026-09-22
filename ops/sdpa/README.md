# sdpa-op: `aten::scaled_dot_product_attention` 多平台实现

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
复用同一语义参考、黄金数据、三层测试与 A1 注册链。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::scaled_dot_product_attention`（SDPA / flash-attention 族） |
| 语义 | `softmax(QKᵀ·scale + mask)·V` · GQA · causal · bool/float mask · fp32 内部 |
| 路线 | **A1** aten 拦截，旧业务代码零改动 |
| 平台 | **ascend910**: 自研 Triton online-softmax；**p800-kunlunxin**: 厂商 efficient-attention 委托 + fp32 精度补偿 |
| 公共入口 | `kernel/triton_level.py` facade → `kernel/backends/{ascend910,p800_kunlunxin}.py` |
| 验证 | P800: kernel 37/37 · 黄金 265/265 · A1 拦截/梯度 · mini-decoder ✅；Ascend 原验证保留 |
| P800 性能 | 强制读回输出的 fp16 采样相对 Python `F.sdpa` 加速 **1.02-1.14x**；FlagGems 2k/4k 比 ours 慢 5.1-6.6x |

平台绑定与坑位见 [PLATFORM.md](PLATFORM.md)，多平台扩展流程见
[MERGE.md](MERGE.md)。

## 快速开始

```bash
# 本机有 torch_xmlir 时自动选 p800；Ascend 机器可显式传 ascend910
python3 example.py p800-kunlunxin

python3 test/kernel_level.py p800-kunlunxin
python3 test/op_level.py p800-kunlunxin
python3 test/framework_level.py p800-kunlunxin
python3 probes/guard_check.py p800-kunlunxin

# CPU 黄金生成后，同一 265 组可直接测任意平台
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl triton --device cuda:1

python3 script/bench_perf.py --device cuda:1 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 script/bench_cross_platform.py --device cuda:1
```

设备映射遵循 `configs/devices/p800-kunlunxin.yaml` 与当前共享镜像的
`CUDA_VISIBLE_DEVICES=1,2`：稳定直测设备为 `cuda:1`。`cuda:0` 上厂商
SDPA 曾挂起，请先不要把它作为默认回归设备。

## 目录要点

```
sdpa/
├── reference.py              # CPU fp32 语义参考
├── register.py               # A1 注册 + autograd.Function
├── _profile.py               # ascend910 / p800-kunlunxin 本地 profile
├── kernel/
│   ├── triton_level.py       # 兼容 facade（旧 import 不变）
│   ├── backends/ascend910.py
│   ├── backends/p800_kunlunxin.py
│   ├── torch_level.py        # ATen 组合对照
│   └── _native_shim.py
├── test/                     # kernel / op / framework 三层
├── goldendata/               # 声明式 265 组黄金
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

## 应用层

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

## Ascend 910 保留能力

Ascend 版仍是完整 Triton 主实现（online-softmax two-pass、causal 截断、
GQA/双 mask/尾块），历史精度、性能与根因分析见
[REPORT.md](REPORT.md)、[reports/development.md](reports/development.md)、
[reports/performance.md](reports/performance.md)。
