#!/usr/bin/env python3
# 多尺度扫描图（消费 perf_sweep.json）:
#   图5: S 扫描折线（每 D 一子图，三方 + 比值轴）
#   图6: D 扫描折线（固定 S=2048）
#   图7: 差距热力图（S×D 网格，颜色=ours/native 比值）
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
C_OURS, C_NAT, C_GEMS = "#2563eb", "#16a34a", "#9ca3af"
ROOT = Path("/root/sdpatten-op")
FIG = ROOT / "reports" / "figs"
rows = json.load(open(ROOT / "reports" / "perf_sweep.json"))
S_LIST = sorted({r["S"] for r in rows})
D_LIST = sorted({r["D"] for r in rows})

def get(S, D, key):
    for r in rows:
        if r["S"] == S and r["D"] == D:
            return r[key]
    return None

# ────────── 图5: S 扫描（每 D 一个子图，上=延迟 下=比值）──────────
fig, axes = plt.subplots(2, len(D_LIST), figsize=(11, 6),
                         sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
for i, D in enumerate(D_LIST):
    ax, ax2 = axes[0][i], axes[1][i]
    ours = [get(S, D, "ours_ms") for S in S_LIST]
    nat = [get(S, D, "native_ms") for S in S_LIST]
    gems = [get(S, D, "gems_ms") for S in S_LIST]
    ax.plot(S_LIST, ours, "o-", color=C_OURS, label="Ours", lw=1.8)
    ax.plot(S_LIST, nat, "s-", color=C_NAT, label="Native", lw=1.8)
    if all(gems):
        ax.plot(S_LIST, gems, "^--", color=C_GEMS, label="FlagGems", lw=1.5)
    ax.set_yscale("log")
    ax.set_title(f"D = {D}", fontsize=10)
    if i == 0:
        ax.set_ylabel("Latency (ms, log)")
        ax.legend(frameon=False, fontsize=8)
    ratio = [o / n for o, n in zip(ours, nat)]
    ax2.plot(S_LIST, ratio, "o-", color=C_OURS, lw=1.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("序列长度 S")
    if i == 0:
        ax2.set_ylabel("Ours/Native")
    ax2.axhline(1.0, color=C_NAT, ls=":", lw=1)
fig.suptitle("多尺度性能扫描: 延迟随 S 变化 (fp16, causal, H=16)  上:绝对延迟  下:与原生比值",
             fontsize=11)
fig.savefig(FIG / "sweep_S.pdf"); fig.savefig(FIG / "sweep_S.png")
plt.close(fig)

# ────────── 图6: D 扫描（S=2048）──────────
fig, ax = plt.subplots(figsize=(6, 3.8))
S0 = 2048
ours = [get(S0, D, "ours_ms") for D in D_LIST]
nat = [get(S0, D, "native_ms") for D in D_LIST]
gems = [get(S0, D, "gems_ms") for D in D_LIST]
x = np.arange(len(D_LIST)); w = 0.26
ax.bar(x - w, ours, w, color=C_OURS, label="Ours")
ax.bar(x, nat, w, color=C_NAT, label="Native")
# gems D=256 编译失败（UB 溢出）——有效点才画，失败位画叉标注
gx = [i for i, gv in enumerate(gems) if gv]
gv_ = [g for g in gems if g]
if gx:
    ax.bar([x[i] + w for i in gx], gv_, w, color=C_GEMS, hatch="//",
           label="FlagGems")
for i, gv in enumerate(gems):
    if not gv:
        ax.text(x[i] + w, 0.02, "✗", ha="center", fontsize=11,
                color="#dc2626", fontweight="bold")
        ax.annotate("编译失败", (x[i] + w, 0.02), fontsize=7,
                    color="#dc2626", ha="center", va="top",
                    textcoords="offset points", xytext=(0, -12))
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([f"D={d}" for d in D_LIST])
ax.set_ylabel("Latency (ms, log)")
ax.legend(frameon=False)
for i, (o, n) in enumerate(zip(ours, nat)):
    ax.text(i - w, o * 1.1, f"{o/n:.1f}x", ha="center", fontsize=8,
            color=C_OURS)
ax.set_title(f"head_dim 扫描 (S={S0}, fp16, causal)", fontsize=11)
fig.savefig(FIG / "sweep_D.pdf"); fig.savefig(FIG / "sweep_D.png")
plt.close(fig)

# ────────── 图7: 差距热力图 S×D ──────────
fig, ax = plt.subplots(figsize=(6.5, 3.6))
M = np.array([[get(S, D, "ours_ms") / get(S, D, "native_ms")
               for D in D_LIST] for S in S_LIST])
im = ax.imshow(M, cmap="YlOrRd", aspect="auto",
               norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(M.max(), 12)))
ax.set_xticks(range(len(D_LIST)))
ax.set_xticklabels([f"D={d}" for d in D_LIST])
ax.set_yticks(range(len(S_LIST)))
ax.set_yticklabels([f"S={s}" for s in S_LIST])
for i in range(len(S_LIST)):
    for j in range(len(D_LIST)):
        ax.text(j, i, f"{M[i,j]:.1f}x", ha="center", va="center",
                fontsize=9, fontweight="bold" if M[i, j] > 8 else "normal",
                color="white" if M[i, j] > 6 else "black")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Ours / Native (log)", fontsize=8)
ax.grid(False)
ax.set_title("与原生 CANN 的差距分布 (S×D 网格, fp16, causal)", fontsize=11)
fig.savefig(FIG / "sweep_heatmap.pdf"); fig.savefig(FIG / "sweep_heatmap.png")
plt.close(fig)

print("生成: sweep_S / sweep_D / sweep_heatmap")
