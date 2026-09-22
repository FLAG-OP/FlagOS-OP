# 跨平台统一基准（bench_cross_platform.py）

> 目的: 同一套测试样例跑 910 / A100（或任意 torch 后端），
> 产出同 schema JSON 直接对比——迁移到 A100 服务器后 clone 即测。

## 快速开始

```bash
# 910（本库主环境）
python3 script/bench_cross_platform.py

# A100 服务器（迁移后）
pip install torch flash-attn        # flash-attn = FA2 论文同款实现
git clone https://github.com/FLAG-OP/FlagOS-OP && cd FlagOS-OP/ops/sdpa
python3 script/bench_cross_platform.py --device cuda:0
# → reports/cross_platform_cuda_0.json
```

## 协议（对齐 FA2 论文官方 benchmark，arXiv:2307.08691）

| 项 | 值 |
|---|---|
| 总 tokens | 16384 恒定（batch = 16k/S） |
| S | 512 / 1k / 2k / 4k / 8k |
| D | 64（32 头）/ 128（16 头），hidden 2048 |
| dtype / mask | fp16 / causal / 前向 |

## 实现自动探测

| key | 条件 | 说明 |
|---|---|---|
| `native` | 恒有 | 各平台 `F.sdpa` 最优后端（910=CANN 闪电注意力, A100=FA2/cuDNN） |
| `flash_attn` | 装了 flash-attn 包 | A100 上的论文同款实现（910 上无此包，自动跳过） |
| `ours_triton` | device ∈ `SUPPORTED_DEVICE_TYPES` | 自研 kernel 为 ascend910 绑定（PLATFORM.md §2）；CUDA 未移植自动跳过并提示 MERGE.md |

## 输出与对比

`reports/cross_platform_<device>.json`，行 schema:
`{S, D, H, B, flops_g, <impl>_ms, <impl>_tflops}`。
两台机器的 JSON 同 schema，直接并排成表或喂给
`script/make_a100_figs.py` 同款图。

## 910 基线（2026-09-22 实测，供 A100 结果对照）

native（CANN）: D=128 S=8k **158.3 TF** / S=4k 149.2 TF / S=2k 122.4 TF
（与 reports/perf_a100.md 图8 同源；利用率口径见该报告注意事项）
ours_triton: D=128 ~6.4 TF 恒定（大 batch 下 64×64 tile 并行度短板
进一步显现——生产路径由 auto_dispatch 截流至 native，见
reports/fusion_vs_dispatch.md）。

## 注意事项

- 共享服务器请选空闲卡（910 侧 ±15µs 扰动）
- A100 侧若 flash-attn 编译困难，`native` 路径（PyTorch 内置 FA2）
  已代表论文水位；两者差异 <5%
- 首次运行含 Triton 编译（910 侧 ~1 分钟），后续命中缓存
