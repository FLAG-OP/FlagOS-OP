# 算子测试报告: detach_

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `detach_` / 自用/实验 / torch 级（别名/元数据（autograd）） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（24 项） |
| 框架层（注册/拦截） | N/A | `test/op_level.py` | detach_ 非 backend dispatch，不可 A1 注册 |
| 注册契约 | ✅ | PR #25 / `register.py` | 签名对齐；仍按设计拒绝 A1 注册 |
| 应用层 | ✅ | `test/framework_level.py` | PASS |
| 黄金 | ✅ | `script/` | 36/36 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 正确性结果

判定：原地返回 self + requires_grad=False + 存储不变。覆盖：4 shape × 3 dtype × {requires_grad True/False}。

## 4. 性能结果

0.0012ms（元数据/同步操作，非门禁项）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无设备码 | 无 Triton 实现 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

适用层 Must 项（原地返回 self + requires_grad=False + 存储不变）**通过**；注册/拦截层为 N/A。
