# 算子测试报告: copy_

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `copy_` / A1 / Triton 级 |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层 | ✅ | `test/op_level.py cambricon` | PASS |
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

Must 项（三层全绿、黄金、原地/广播/cast 正确）**通过**。
