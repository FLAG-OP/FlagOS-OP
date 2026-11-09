# 算子总体报告：aten.sigmoid

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`；见 [reference.py](reference.py) |
| 路线 / 级别 | A1 / Triton |
| 设备 | P800 物理 XPU 2；单卡 `p800-liyanz-local`，软件锁等价于 `p800-kunlunxin-gems5` |
| 日期 / 状态 | 2026-11-09 / 实现与自测完成（C1） |

## 实现矩阵

| 级别 | 状态 | 文件 |
|---|---|---|
| torch | 完成 | [kernel/torch_level.py](kernel/torch_level.py) |
| Triton | 完成并通过 P800 真机验证 | [kernel/triton_level.py](kernel/triton_level.py) |
| 硬件级 | 未实现，本次范围外 | [kernel/hardware_level/BUILD.md](kernel/hardware_level/BUILD.md) |

## 验证矩阵

| 层级 | 结果 | 关键证据 |
|---|---|---|
| kernel | PASS | 75 例、19 shape、3 优化 dtype、3 回退 dtype；误差 0；哨兵通过 |
| A1 op | PASS | 9 次拦截，counter=9；布局和 dtype 回退通过 |
| C1 应用 | PASS | Transformer 激活 `2×64×4096`，fp16/bf16/fp32，3/3 |
| 黄金 | PASS | 93/93；最差 abs `1.192e-07`、rel `1.680e-07` |
| 性能矩阵 | 完成 | 4 shape × 3 dtype × 3 实现 = 36 个独立子进程 |
| dispatch | 完成 | 直调 `47.970 μs`，注册 `55.877 μs`，增量 `7.908 μs` |
| 中央性能入口 | PASS | `ops.sigmoid.triton=0.058 ms`；无历史 baseline，NEW，0 FAIL |

## 性能结论

- 正式 FlagGems 5.0 环境中，自研 12 个组合均未超过原生。
- 最接近原生的是 `2048×2048 bf16`：自研 `0.077650 ms`、原生 `0.067039 ms`，比值 `1.158×`。
- 小尺寸受约 49～58 μs 自研启动固定成本影响，最差为 `1×14336 bf16` 的 `4.276×`。
- 自研仅在大张量 bf16/fp32 两项略快于 FlagGems 5.0，原生仍是当前最佳基线。

## 结论

P800 forward 的功能、精度、A1 集成、C1 消费和性能基线证据完整，可以进入代码评审。当前价值是建立正确、可迁移的实现和应用边界，不声称普遍性能提升。小尺寸原生路由或 kernel 启动优化必须在后续提交中以独立证据验证。

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 应用范围 | [reports/scope.md](reports/scope.md) |
