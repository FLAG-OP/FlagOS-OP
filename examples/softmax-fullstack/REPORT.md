# 算子总体报告: softmax

| 项 | 值 |
|---|---|
| 算子名称 | `softmax`（最后一维） |
| 语义 | [reference.py](reference.py)（fp32 计算后 cast 回输入 dtype） |
| 目标设备 / 路线 | `p800-kunlunxin` / **A1**（aten `_softmax` 拦截） |
| 日期 / 状态 | 2026-08-28 / **定稿（教学样例）** |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | ATen `F.softmax`（原生对照实现） |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ | 自研三遍流式归约，尾块 pad + 固定 BLOCK_N |
| 硬件级 | [kernel/hardware_level/](kernel/hardware_level/README.md) | ⬜ 置空 | 厂商库无同语义原语，XTC 未提供 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| kernel 层 | `test/kernel_level.py`（24 组精度含尾块回归 + 哨兵 + 三方性能） | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py`（aten 拦截 + 拦截路径精度） | ✅ | — |
| 应用层 | `test/framework_level.py`（attention scores 消费） | ✅ | — |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ | [goldendata/](goldendata/README.md) |
| 性能 | perf 基线门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 自研 Triton | FlagGems | 原生 ATen |
|---|---|---|---|
| fp32 精度 | **2.4e-7** | ⚠ 1.5e-2（波动至 4e-2） | 0 |
| 1024² bf16 延迟 | 0.060ms | 0.028ms | **0.010ms** |

## 结论与遗留

三层全绿、精度三方最优；性能落后原生 6x、落后 FlagGems 2x——
FlagGems 的快以精度超差为代价（fp32 误差 5 个数量级），自研路线
正确性优先。后续优化方向: 单遍在线归约（Flash 风格），目标在
不降精度下逼近 0.028ms。本样例同时是 **#15（尾块归约污染 +
autotuner 非法 num_warps）的发现与修复载体**。
