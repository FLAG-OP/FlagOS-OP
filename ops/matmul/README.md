# matmul-op: `aten::matmul` 平头哥真武 810E（thead-ppu）实现

> 按 FlagOS-OP 算子模板交付（issue #55）。通用矩阵乘（台账 41,595 次/285 模型; 公共骨架, PPU 首交付）。
> 当前硬件平台: thead-ppu（PPU-ZW810E ×16, CUDA 兼容层直曝 cuda:N）；
> C2 真实 vLLM 证据归 thead-ppu-c2 栈 profile 承接（待排期）。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::matmul` |
| 平台 | thead-ppu（PPU-ZW810E; torch 2.10.0 / triton 3.6.0） |
| 路线 | A1 · native 委托 |
| 生产实现 | 原生委托（分形态组合反向: 2D/ND/广播/1D 全覆盖） |
| 关键结论 | 8 形态×3 dtype 5e-2 rel 内；132 TFLOPS；分形态反向 grad diff=0 |
| 验证 | 三层全绿 + 黄金 native: matmul: native @ cuda:6 30 / 30 PASS, worst=7.692e-03 + framework 消费取证（2026-11-25/26） |

## 快速开始

```bash
cd ops/matmul
python3 example.py thead-ppu                       # 三层一键
python3 script/gen_golden.py --device cpu          # 黄金生成
python3 script/check_accuracy.py --impl native --device cuda:6
python3 script/bench_perf.py --device cuda:6 --json-out reports/perf.json
```

平台绑定（redispatch 键/注册键/平台分化）见 [PLATFORM.md](PLATFORM.md)；
完整验证矩阵与证据见 [REPORT.md](REPORT.md)。
