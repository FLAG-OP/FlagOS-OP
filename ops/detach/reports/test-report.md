# 算子测试报告: detach

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `detach` / A1 / torch 级（别名/元数据（autograd）） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（25 项） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | aten::detach@PrivateUse1（拦截 1 次） |
| 注册契约 | ✅ | PR #25 / `register.py` | `counter` + `platform`；canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`；包态导入优先 |
| 应用层 | ✅ | `test/framework_level.py` | PASS |
| 黄金 | ✅ | `script/` | 36/36 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 正确性结果

判定：共享存储 + requires_grad=False + shape/dtype/stride。覆盖：4 shape × 3 dtype × {requires_grad True/False} + 非连续 view。

## 4. 性能结果

0.0066ms（元数据/同步操作，非门禁项）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无设备码 | 无 Triton 实现 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

Must 项（适用层验证通过（C1 等价 torch API）、共享存储 + requires_grad=False + shape/dtype/stride）**通过**。

## 7. P800（p800-kunlunxin）复验（2026-10-26）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `python3 test/kernel_level.py p800-kunlunxin` | PASS |
| 框架层 | `python3 test/op_level.py p800-kunlunxin` | PASS；拦截 1 次 |
| 应用层 | `python3 test/framework_level.py p800-kunlunxin` | PASS（等价 torch API 子进程） |
| 黄金 | `python3 script/check_accuracy.py --impl torch --device cuda:1` | 36/36，0 fail |
| 性能门禁 | `python3 scripts/perf_compare.py --device p800-kunlunxin` | OK |

结论：P800 复验 **通过**。共享存储的非原地视图，P800 直接复用。
