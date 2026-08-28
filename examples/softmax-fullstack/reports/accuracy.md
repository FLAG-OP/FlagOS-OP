# 精度报告: softmax（已填，2026-08-28 P800 实测）

## 1. 用例覆盖

| case | shape(s) | dtype | 分布 | special | 组数 |
|---|---|---|---|---|---|
| small | 64×1024, 128×512 | bf16/fp16/fp32 | normal σ3 | — | 9 |
| tail | 8×5000, 16×5120, 32×3072 | bf16/fp32 | normal σ3 | — | 18 |
| extreme | 128×5120 | fp32 | uniform×8 | zeros, large | 3 |

黄金由 `script/gen_golden.py` 按规格生成（sha256 索引）。

## 2. 结果（自研 Triton，容差 fp32=1e-5 / bf16·fp16=1e-2）

| dtype | 自研 max err | FlagGems | 原生 |
|---|---|---|---|
| bf16 | 1.8e-3 ✓ | ~2.0e-2 ⚠ | 1.9e-3 ✓ |
| fp16 | 2.2e-4 ✓ | ~2.0e-2 ⚠ | 2.3e-4 ✓ |
| fp32 | **2.4e-7** ✓ | **1.5e-2~4e-2 ⚠** | 0 ✓ |

哨兵: 确定性 ✓ · 输入敏感 ✓。

## 3. 失败分析（历史，已修复）

| case | 现象 | 根因 | 关联 |
|---|---|---|---|
| N=5120/3072 | err 0.6 / inf | 尾块 masked load + `tl.sum` 污染（`other`/`tl.where` 三种防护无效） | [#15a](../../../docs/known-issues.md) |
| 同 kernel 时对时错 | 不可复现 | autotuner 选出配置表外的 `num_warps=5` | [#15b](../../../docs/known-issues.md) |

修复: wrapper pad 到 2048 整倍数 + 固定 BLOCK_N=2048；测试形状补
非整倍数 N 堵住漏测缺口。
