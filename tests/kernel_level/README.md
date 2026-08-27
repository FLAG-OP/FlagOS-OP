# kernel 层测试

不经任何注册/分发，直测 kernel 本体——厂商算子注册路线的核心验证层。

每路线一个测试模块，统一签名 `run(profile)`，返回 `bool` 或
`{"ok": bool, ...指标}`（精度/哨兵/性能会落盘到结果 JSON）:

| 模块 | 内容 |
|---|---|
| [test_a1.py](test_a1.py) | Triton gelu 直测（精度/哨兵/性能） |
| [test_a2.py](test_a2.py) | Triton gelu_and_mul 直测 |
| [test_b.py](test_b.py) | 厂商 kernel 清单直测 + csrc JIT 闭环 + 哨兵负例 |
| [test_consistency.py](test_consistency.py) | 跨层一致性（算子库层 ↔ 框架层） |

断言体系见 [docs/testing.md](../../docs/testing.md)。
性能用例（`perf_cases`）挂入[性能回归](../../docs/performance-regression.md)。
