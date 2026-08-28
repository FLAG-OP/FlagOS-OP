# 算子样板（copy-paste 起点）

开发一个新算子，从这里开始:

```bash
cp -r templates/operator <你的算子名>/
cd <你的算子名>/
# 逐个把 <TODO> 替换为你的实现
```

## 文件清单

| 文件 | 作用 | 对应全链路步骤 |
|---|---|---|
| [reference.py](reference.py) | fp32 语义参考（判卷标准，先写它） | 第 0 步 |
| [kernel.py](kernel.py) | Triton kernel（含 device 上下文与尾块安全两条硬约束） | 第 1 步 |
| [test_kernel.py](test_kernel.py) | kernel 层直测: 精度/哨兵/性能 + metrics 落盘 | 第 2 步 |
| [register.py](register.py) | 三条路线注册（A1 aten / A2 dispatch / B vendor，按需删） | 第 3 步 |
| [test_op.py](test_op.py) | op 层注册/钉选/拦截验证骨架 | 第 3 步 |
| [test_framework.py](test_framework.py) | 应用层真实推理验证骨架（复用库 harness） | 第 4 步 |
| [perf.py](perf.py) | 性能回归用例（挂入 perf 基线） | 第 6 步 |

## 开发顺序

1. **先写 `reference.py`**——语义没定清楚之前不要写 kernel
2. 实现 `kernel.py`，跑 `python3 test_kernel.py --device <profile>` 直到全绿
3. 选一条路线（`register.py` 里三选一），跑 `test_op.py`
4. 应用层验证走 `test_framework.py`（复用本库 harness，见[测试体系](../../docs/testing.md#framework)）
5. `perf.py` 挂入 `common/perf_registry.py`，进回归门禁
6. 用 [测试报告](../op-test-report.md) / [开发报告](../op-development-report.md) 模板收尾，
   样例见 [reports/examples](../../reports/examples/)

## 三条硬约束（本库实测踩坑，违反必错）

1. **Triton 启动必须包 device 上下文**，否则首次后的启动静默 no-op
   （[known-issues #11](../../docs/known-issues.md)）
2. **尾块归约要安全**：N 非 BLOCK 整数倍时 masked load + `tl.sum` 会被污染
   （`other`/`tl.where` 均救不了）；pad 到整倍数或改两阶段归约
   （[known-issues #15](../../docs/known-issues.md)）
3. **不要用 `@triton.autotune`**：本栈 autotuner 会选出非法 `num_warps=5`，
   同一 kernel 不可复现地时对时错（#15b）；固定 BLOCK 配置
