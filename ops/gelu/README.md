# gelu-op: `aten::gelu` 平头哥真武 810E（thead-ppu）实现

> 按 FlagOS-OP 算子模板交付（issue #55）。GELU（none/tanh 双模式）+ 解析 backward；FlagFix #5867 迁移记录。
> 当前硬件平台: thead-ppu（PPU-ZW810E ×16, CUDA 兼容层直曝 cuda:N）；
> C2 真实 vLLM 证据归 thead-ppu-c2 栈 profile 承接（待排期）。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::gelu` |
| 平台 | thead-ppu（PPU-ZW810E; torch 2.10.0 / triton 3.6.0） |
| 路线 | A1 · Triton 自研 |
| 生产实现 | Triton 自研（#11 device 上下文 / #15 不适用[无归约] / geluFIX 写法规约） |
| 关键结论 | 自研 Triton 1.088x 原生（0.131ms vs 0.143ms @ 8192² fp16, 2.05 TB/s） |
| 验证 | 三层全绿 + 黄金 triton: [gelu:triton@cuda:6] 150/150 PASS, worst=9.766e-04 + framework 消费取证（2026-11-25/26） |

## 快速开始

```bash
cd ops/gelu
python3 example.py thead-ppu                       # 三层一键
python3 script/gen_golden.py --device cpu          # 黄金生成
python3 script/check_accuracy.py --impl triton --device cuda:6
python3 script/bench_perf.py --device cuda:6 --json-out reports/perf.json
```

平台绑定（redispatch 键/注册键/平台分化）见 [PLATFORM.md](PLATFORM.md)；
完整验证矩阵与证据见 [REPORT.md](REPORT.md)。
