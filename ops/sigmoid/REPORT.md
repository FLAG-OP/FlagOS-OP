# 算子总体报告：aten.sigmoid

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`；见 [reference.py](reference.py) |
| 路线 / 级别 | A1 / Triton |
| 设备 | P800 物理 XPU 2；单卡 `p800-liyanz-local`，软件锁等价于 `p800-kunlunxin-gems5` |
| 日期 / 状态 | 2026-11-22 / 255 边界修复完成，最终硬件复测待环境恢复 |

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
| C2 vLLM | 待复测 | 真实 EngineCore 等价消费者脚本已具备；当前 P800 运行时阻塞 |
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

表中 75/9/3、93/93、36/36 为 2026-11-09 修改前历史基线。2026-11-22 当前代码已通过环境锁 9/9、182 项基础单测、17/17 包化替代扫描、CPU 黄金 reference/torch 各 93/93，以及 255 专用路径 3 dtype × 2 shape 的代表点真机检查。普通 `1×14336` Triton 烟测同样超时，表明当前阻塞不限于 255；修改后的三层、Triton 黄金、C2 与性能矩阵仍待运行时恢复后复测。因此当前可继续代码评审，但不能标记为最终硬件验收完成。

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 应用范围 | [reports/scope.md](reports/scope.md) |
