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
| kernel 层（75 例） | PASS | 0 | 0 |
| 黄金回归（93 例） | PASS | `1.192e-07` | `1.680e-07` |

fp16、bf16 输出与 fp32 参考转换结果逐位一致；float32 最差误差远低于容差。确定性、输入敏感性、NaN mask、空张量和三类非连续布局均通过。
