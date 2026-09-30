# 算子总体报告: embedding（aten::embedding）

| 项 | 值 |
|---|---|
| 算子名称 | `aten::embedding`（稠密查表 + dense 反向） |
| 语义 | [reference.py](reference.py)（`weight[index]` 任意 index 形状；padding/scale_grad/sparse 不改前向） |
| 硬件平台 | **2 个**：p800-kunlunxin（厂商委托）· **ascend910**（本 PR） |
| 路线 | **A1**（aten 拦截） |
| 日期 / 状态 | 2026-09-30 / 定稿（第二平台） |

## 实现矩阵（每平台）

| 平台 | 级别 | 文件 | 生产路径 | 状态 |
|---|---|---|---|---|
| p800-kunlunxin | 厂商委托 / Triton 探针 | kernel/p800_kunlunxin.py · kernel/triton_level.py | `aten::index_select` 委托 | ✅ |
| **ascend910** | 厂商委托 | **kernel/ascend910.py** | `aten::index_select`（CANN 行采集，实测 0.185ms@131k×4k×16k 与 F.embedding 同源同速）· dense 反向委托 `aten::embedding_dense_backward` + scale_grad_by_freq 逐出现逆频率缩放 | ✅ |

Triton gather 在两平台均慢数十倍（P800 4.6-69x、Ascend 探针见
reports/performance.md），保留为可复现平台探针——embedding 的本质是
gather，行采集是厂商库强项，**生产委托是两平台一致的工程结论**。

## 验证矩阵（ascend910 本 PR 实测）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `EMBEDDING_PROFILE=ascend910 python3 test/kernel_level.py` | ✅ 20 前向 + 6 反向 case，err=0.0（委托原生本征精确） |
| op 层 | `test/op_level.py` | ✅ 拦截 5 次 · 注册=直调逐位 · dense/scale_freq 梯度 err=0.0 · sparse bwd 守卫 |
| 应用层 | `test/framework_level.py` | ✅ nn.Embedding+TokenMLP 消费: logits diff=0 · top1=1.0 · 续写一致率 1.0 |
| 黄金 | `gen_golden.py` + `check_accuracy.py --impl ascend` | ✅ 350 PASS（data 本地生成） |
| 一键 | `EMBEDDING_PROFILE=ascend910 python3 example.py` | ✅ 三层全绿 |

## 性能速览（fp16）

| shape | ours（index_select） | native F.embedding | Triton 探针 |
|---|---|---|---|
| prefill_16k_d128 | 0.081ms | 0.060ms | 1.45ms（18x 慢） |
| long_131k_d128 | 0.152ms | 0.095ms | 10.4ms（69x 慢） |
| backward 16k | 0.197ms | 0.192ms | —（0.98x） |

完整数据: reports/perf_fp16_ascend910.json。

## A1 注册点（平台差异，sdpa 开发报告 §3.1 证据链复用）

torch_npu 栈拦截点为 **AutogradPrivateUse1**（PrivateUse1 永不命中）；
register_a1 已按 dispatch_key 自动选择 backend（p800→AutogradCUDA /
ascend→AutogradPrivateUse1），或显式 `platform="ascend910"`。

## 结论

第二平台按 MERGE 流程合入: 生产路径与 P800 同构（厂商委托），
三层 + 黄金 + 性能全绿，委托实现 err=0.0。Triton 探针数据进一步
佐证两平台一致的结论——embedding 峰值属厂商库，Triton 价值在
长尾与可移植（若未来需要自定义 padding 语义/量化查表再启用）。
