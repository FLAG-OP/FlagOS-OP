# 算子测试报告: result_type

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `result_type` / 自用/实验 / torch 级（dtype 元数据） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（64 项） |
| 框架层（注册/拦截） | N/A | `test/op_level.py` | result_type 为纯函数，不可 A1 注册 |
| 注册契约 | ✅ | PR #25 / `register.py` | 签名对齐；仍按设计拒绝 A1 注册 |
| 应用层 | ✅ | `test/framework_level.py` | PASS |
| 黄金 | ✅ | `script/` | 8/8 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 正确性结果

判定：公共 dtype 相等。覆盖：8 dtype × 8 dtype 提升矩阵。

## 4. 性能结果

0.0013ms（元数据/同步操作，非门禁项）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无设备码 | 无 Triton 实现 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

适用层 Must 项（公共 dtype 相等）**通过**；注册/拦截层为 N/A。

## 7. P800（p800-kunlunxin）复验（2026-10-26）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `python3 test/kernel_level.py p800-kunlunxin` | PASS |
| 框架层 | `python3 test/op_level.py p800-kunlunxin` | PASS；按设计不注册 |
| 应用层 | `python3 test/framework_level.py p800-kunlunxin` | PASS（等价 torch API 子进程） |
| 黄金 | `python3 script/check_accuracy.py --impl torch --device cuda:1` | 7/7（1 skip），0 fail |
| 性能门禁 | `python3 scripts/perf_compare.py --device p800-kunlunxin` | OK |

结论：P800 复验 **通过**。P800 无 fp64，float64 黄金用例按设备能力 skip（7 pass / 0 fail / 1 skip）。
