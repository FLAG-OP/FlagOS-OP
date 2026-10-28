# 算子测试报告: copy_

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `copy_` / A1 / Triton 级 |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层 | ✅ | `test/op_level.py cambricon` | PASS |
| 注册契约 | ✅ | PR #25 / `register.py` | `counter` + `platform`；canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`；包态导入优先 |
| 应用层 | ✅（等价） | `test/framework_level.py cambricon` | PASS |
| 黄金回归 | ✅ | `script/` 两脚本 | 78/78 |
| 性能门禁 | ✅ | `scripts/perf_run.py --pattern ops.copy_` | 可建基线 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 精度结果

| 类别 | dtype | max err | 原地返回 | 判定 |
|---|---|---|---|---|
| 同形 | fp32/fp16/bf16 交叉 | 0 | ✅ | ✅ |
| cast | fp32↔fp16↔bf16 | 0（含合法 ±inf） | ✅ | ✅ |
| 广播 | fp32/bf16 | 0 | ✅ | ✅ |

黄金 78 组全通过；非连续 dst/src、自拷贝别名、int32 回退均正确。
哨兵: 确定性 ✅ 敏感 ✅。

## 4. 性能结果

8192² fp32: 自研 0.242ms / 原生 0.237ms（差 ~2%）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 全局注册 copy_ 侵入性强 | 编排需子进程隔离 | 已注明 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

Must 项（适用层验证通过（C1 等价 torch API）、黄金、原地/广播/cast 正确）**通过**。

## 7. P800（p800-kunlunxin）复验（2026-10-26）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `python3 test/kernel_level.py p800-kunlunxin` | PASS |
| 框架层 | `python3 test/op_level.py p800-kunlunxin` | PASS；拦截 4 次 |
| 应用层 | `python3 test/framework_level.py p800-kunlunxin` | PASS（等价 torch API 子进程） |
| 黄金 | `python3 script/check_accuracy.py --impl triton --device cuda:1` | 78/78，0 fail |
| 性能门禁 | `python3 scripts/perf_compare.py --device p800-kunlunxin` | OK |

结论：P800 复验 **通过**。dst=bf16 且 src 非 bf16 时先用原生 type_as 转换；P800 原生 copy_ 广播路径实测错值，reference 已物化广播源后判卷。
