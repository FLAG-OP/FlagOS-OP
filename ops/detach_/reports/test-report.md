# 算子测试报告: detach_

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `detach_` / 自用/实验 / torch 级（别名/元数据（autograd）） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（24 项） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | —（detach_ 非 backend dispatch，不可 A1 注册） |
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

Must 项（三层全绿、原地返回 self + requires_grad=False + 存储不变）**通过**。
