#!/usr/bin/env python3
# A100 对标图: FA2 论文公开数据 vs 本机实测（同协议、利用率归一化）
# 图8: 绝对 TFLOPS 对比（A100-FA2 / 910-native / 910-ours）
# 图9: 峰值利用率归一化（跨平台可比口径）
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
})

ROOT = Path("/root/sdpatten-op")
FIG = ROOT / "reports" / "figs"
rows = json.load(open(ROOT / "reports" / "perf_fa2_protocol.json"))

# A100 80GB SXM FA2 前向（arXiv:2307.08691 fig.4 口径读取，causal）
# D=128 曲线在 1k-8k 段约 185-235 TFLOPS; D=64 约 120-190
A100 = {64: {512: 120, 1024: 155, 2048: 175, 4096: 185, 8192: 190},
        128: {512: 160, 1024: 200, 2048: 220, 4096: 230, 8192: 235}}
# 峰值分母: A100 80GB SXM fp16 = 312 (官方); Ascend910 官方 spec 256
# (32 核口径; 本机 910_9382 为 24 核, 实际峰值可能更低 → 利用率为保守下限)
A100_PEAK, NPU_PEAK = 312.0, 256.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, D in zip(axes, (64, 128)):
    sub = [r for r in rows if r["D"] == D]
    S = [r["S"] for r in sub]
    ax.plot(S, [A100[D][s] for s in S], "s--", color="#dc2626",
            label="A100 80GB · FA2 (论文公开)", lw=1.8)
    ax.plot(S, [r["native_tflops"] for r in sub], "o-", color="#16a34a",
            label="Ascend910 · CANN 原生 (实测)", lw=1.8)
    ax.plot(S, [r["ours_tflops"] for r in sub], "^-", color="#2563eb",
            label="Ascend910 · 本实现 Triton (实测)", lw=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("序列长度 S (batch=16k/S)")
    ax.set_ylabel("前向有效算力 (TFLOPS)")
    ax.set_title(f"D = {D}", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("FA2 论文协议对标: Ascend910 vs A100 (fp16, causal, 前向)",
             fontsize=12)
fig.savefig(FIG / "a100_compare.pdf"); fig.savefig(FIG / "a100_compare.png")
plt.close(fig)

# 图9: 峰值利用率（各自硬件峰值归一）
fig, ax = plt.subplots(figsize=(7, 4))
labels, vals, colors = [], [], []
for D in (64, 128):
    sub = [r for r in rows if r["D"] == D and r["S"] == 8192]
    r = sub[0]
    labels += [f"A100·FA2\nD={D}", f"910·CANN\nD={D}", f"910·ours\nD={D}"]
    vals += [A100[D][8192] / A100_PEAK * 100,
             r["native_tflops"] / NPU_PEAK * 100,
             r["ours_tflops"] / NPU_PEAK * 100]
    colors += ["#dc2626", "#16a34a", "#2563eb"]
bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
            ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("峰值利用率 (%)")
ax.set_title("硬件峰值利用率归一化对比 (S=8192, fp16, causal, 前向)", fontsize=11)
fig.savefig(FIG / "peak_util.pdf"); fig.savefig(FIG / "peak_util.png")
plt.close(fig)
print("生成: a100_compare / peak_util")
