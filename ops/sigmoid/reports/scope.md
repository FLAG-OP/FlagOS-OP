# aten.sigmoid 应用范围

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

本文区分自研优化、原生回退和范围外接口，避免把单一平台结果外推。

## 1. 接口范围

| 接口 | 状态 | 说明 |
|---|---|---|
| `aten::sigmoid(Tensor self) -> Tensor` | 自研 forward | 本次交付目标 |
| `aten::sigmoid_` | 范围外 | 原地语义需独立别名分析和注册 |
| `aten::sigmoid.out` | 范围外 | out 参数需独立实现 |
| `aten::sigmoid_backward` | 范围外 | 当前项目面向推理 forward |

## 2. dtype 和布局

| dtype/layout | 路径 | 证据 |
|---|---|---|
| fp16、bf16、fp32 strided | P800 自研 pointwise | 全正确性和性能矩阵 |
| int32、int64、bool | ATen redispatch | float32 输出与原生一致 |
| fp64、complex、非 strided | ATen redispatch | 不声明自研优化 |
| sparse、quantized | 范围外 | 未纳入当前 LLM 推理范围 |

非连续 strided 输入会先连续化，输出数值、shape 和 dtype 保持一致，但不承诺保留输入 stride。

## 3. 已验证尺寸

- 0D 标量、空张量、1D/2D/3D/4D；
- 极小输入 `1`、`17`；
- `255/256/257`、`511/512/513`；
- `65535/65536/65537`；
- 长尾 `1×14336`、`8×5000`、`7×4097`；
- Transformer `2×128×768`、`2×64×4096`、`1×4×32×512`；
- contiguous、转置、步长切片、expand；
- `-Inf/+Inf/NaN/-0/+0` 和饱和值。

性能矩阵覆盖 `1×14336`、`64×1024`、`2×64×4096` 和 `2048×2048`，每项运行 fp16/bf16/fp32，并与 FlagGems 和原生实现比较。

## 4. 性能应用边界

正式 FlagGems 5.0 环境中，自研/原生比值为 `1.158–4.276×`，没有已测组合快于原生。因此本版本不能作为默认全局加速路径；它提供正确的 A1 自研实现和后续优化基线。若用于实际模型，应在上层保留原生路径，直到路由阈值或新 kernel 通过独立性能与精度复测。

## 5. 平台范围

| 平台 | 状态 |
|---|---|
| 昆仑芯 P800 | 已完成本页 C1 forward 验证 |
| 华为昇腾、寒武纪、平头哥、燧原、壁仞 | 未验证，不能由 P800 结果外推 |

通用 Triton fallback 只表示代码结构可移植；平台必须完成真实硬件测试、环境锁、三层验证和性能报告后才能登记支持。

## 6. 验收入口

```bash
python ops/sigmoid/example.py p800-liyanz-local
python ops/sigmoid/script/gen_golden.py --device cpu
python ops/sigmoid/script/check_accuracy.py --impl triton --device p800-liyanz-local
python ops/sigmoid/script/bench_matrix.py --profile p800-liyanz-local --warmup 20 --iters 100
```
