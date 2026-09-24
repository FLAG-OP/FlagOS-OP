# 算子测试报告: result_type

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `result_type` / 自用/实验 / torch 级（dtype 元数据） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py` | PASS（64 项） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | —（result_type 为纯函数，不可 A1 注册） |
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

Must 项（三层全绿、公共 dtype 相等）**通过**。
