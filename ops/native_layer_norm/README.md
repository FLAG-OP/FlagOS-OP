# native_layer_norm-op: `aten::native_layer_norm` 平头哥真武 810E（thead-ppu）实现

> 按 FlagOS-OP 算子模板交付（issue #55）。LayerNorm 私有原语（台账 13,593 次; 与 layer_norm 配对验证）。
> 当前硬件平台: thead-ppu（PPU-ZW810E ×16, CUDA 兼容层直曝 cuda:N）；
> C2 真实 vLLM 证据归 thead-ppu-c2 栈 profile 承接（待排期）。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::native_layer_norm` |
| 平台 | thead-ppu（PPU-ZW810E; torch 2.10.0 / triton 3.6.0） |
| 路线 | A1 · native 委托 |
| 生产实现 | 原生委托（3 返回值; native_layer_norm_backward 反向） |
| 关键结论 | 3 返回值对拍; mean/rstd keepdim 形状平台分化如实登记 |
| 验证 | 三层全绿 + 黄金 native: native_layer_norm: native @ cuda:6 24 / 24 PASS, worst=4.883e-04 + framework 消费取证（2026-11-25/26） |

## 快速开始

```bash
cd ops/native_layer_norm
python3 example.py thead-ppu                       # 三层一键
python3 script/gen_golden.py --device cpu          # 黄金生成
python3 script/check_accuracy.py --impl native --device cuda:6
python3 script/bench_perf.py --device cuda:6 --json-out reports/perf.json
```

平台绑定（redispatch 键/注册键/平台分化）见 [PLATFORM.md](PLATFORM.md)；
完整验证矩阵与证据见 [REPORT.md](REPORT.md)。
