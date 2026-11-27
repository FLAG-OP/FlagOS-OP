# aten.sigmoid 精度报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 覆盖项 | 内容 |
|---|---|
| 环境 | P800 / FlagGems 5.0 官方软件锁 |
| dtype | 自研：fp16、bf16、fp32；回退：int32、int64、bool |
| shape | 0D～4D、空、19 个尺寸、launch 边界、长尾、Transformer |
| layout | contiguous、转置、步长切片、expand |
| 特殊值 | ±Inf、NaN、±0、-100、-20、-1、1、20、100 |
| 容差 | fp32 `2e-6`；fp16 `1e-3`；bf16 `8e-3` |

## 结果

| 测试 | 结果 | 最差 abs | 最差 rel |
|---|---|---:|---:|
| kernel 层（75 例） | 2026-11-27 当前 P800 PASS | 0 | 0 |
| 黄金回归（93 例） | 2026-11-27 当前 Triton 93/93 | `1.192e-07` | `1.680e-07` |

当前 fp16、bf16 输出与 fp32 参考转换结果逐位一致，float32 最差误差远低于 `2e-6` 容差。当前提交同时保留 CPU reference/torch 各 93/93 回归。

## 2026-11-27 当前提交复测

- 当前代码新增连续 `numel=255` 的固定 256 lane 无 mask 路径；第 256 lane 将读取下标钳制到 254，输出裁剪回原 shape。
- 19 个 shape 覆盖 `255/256/257`、长尾、Transformer、空张量和 0D～4D；fp16/bf16/fp32 当前 kernel 75 例全部通过。
- CPU 黄金已重新生成，reference 与 torch 路径均为 93/93、误差 0。
- 当前 P800 Triton 黄金 93/93；最差 abs `1.192e-07`、rel `1.680e-07`。
- 当前 A1 op 9 次命中且计数一致，C1 fp16/bf16/fp32 3/3 均命中并与 baseline 一致。
- 原始日志、提交和镜像摘要见[当前验收材料](../evidence/p800-current-validation-2026-11-27/ANALYSIS.md)。
