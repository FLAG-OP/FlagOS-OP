# 算子测试报告: dropout

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `dropout` / A1 / Triton 级（Philox） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过**（随机分支按统计判定） |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层 | ✅ | `test/op_level.py cambricon` | PASS |
| 应用层 | ✅（等价） | `test/framework_level.py cambricon` | PASS |
| 黄金 | ✅ | `script/` 两脚本 | 189/189 |
| 性能门禁 | ✅ | `scripts/perf_run.py --pattern ops.dropout` | 可建基线 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 精度/正确性结果

| 分支 | 判据 | 结果 |
|---|---|---|
| train=False / p=0 | 返回 input 别名 | ✅ |
| p=1 | 全零 | ✅ |
| p∉[0,1] | 抛错 | ✅ |
| 0<p<1 | 值域 {0, x/(1-p)}、drop 比例、seed 可控 | ✅ 最大 drop 偏差 0.0016 |
| channels_last | 保持内存格式 | ✅ |

黄金 189 组（108 确定性位级 + 81 随机结构统计）全通过。

## 4. 性能结果

8192² fp16 p=0.5: 自研 2.785ms / FlagGems 2.960ms / 原生 0.563ms。
RNG 受限，比原生慢 ~5x，但与 FlagGems 同级。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | Philox RNG 慢 | 性能 | 机制限制 |
| 2 | 随机流与原生不可对齐 | 无（dropout 随机） | 已注明 |
| 3 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

Must 项（三层全绿、确定性分支位级、随机分支统计合规、seed 可控）**通过**。
