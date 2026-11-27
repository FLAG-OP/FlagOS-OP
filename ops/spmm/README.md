# spmm-op: `aten::_sparse_mm` 平头哥真武 810E（thead-ppu）实现

> 按 FlagOS-OP 算子模板交付（issue #57）。CSR × dense（torch.sparse.mm 入口; Gowalla 训练链路消费位）。
> 当前硬件平台: thead-ppu（PPU-ZW810E ×16, CUDA 兼容层直曝 cuda:N）；
> C2 真实 vLLM 证据归 thead-ppu-c2 栈 profile 承接（待排期）。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::_sparse_mm` |
| 平台 | thead-ppu（PPU-ZW810E; torch 2.10.0 / triton 3.6.0） |
| 路线 | A1 · index_add 组合自建 |
| 生产实现 | index_add 组合自建（COO gather + fp32 累加; COO 组合反向） |
| 关键结论 | 原生 cusparse 缺口实锤（冷启动失败/fp16 缺口/C++ abort）→ 自建；fp64 独立判卷 worst 8.5e-3 rel；GCN 消费 8 次 |
| 验证 | 三层全绿 + 黄金 native: spmm: native @ cuda:6 24 / 24 PASS, worst=9.533e-04 + framework 消费取证（2026-11-25/26） |

## 快速开始

```bash
cd ops/spmm
python3 example.py thead-ppu                       # 三层一键
python3 script/gen_golden.py --device cpu          # 黄金生成
python3 script/check_accuracy.py --impl native --device cuda:6
python3 script/bench_perf.py --device cuda:6 --json-out reports/perf.json
```

平台绑定（redispatch 键/注册键/平台分化）见 [PLATFORM.md](PLATFORM.md)；
完整验证矩阵与证据见 [REPORT.md](REPORT.md)。
