# 算子测试报告: detach

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `detach` / A1 / torch 级（别名/元数据（autograd）） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（25 项） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | aten::detach@PrivateUse1（拦截 1 次） |
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

Must 项（三层全绿、共享存储 + requires_grad=False + shape/dtype/stride）**通过**。
