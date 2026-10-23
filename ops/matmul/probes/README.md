# matmul 诊断工具

正式前向和反向入口在 `kernel/triton_level.py`，统一使用动态 K 的 dot
kernel。此目录保留开发时的对照实现与编译排查工具，不参与 A1 注册。

| 文件 | 用途 |
|---|---|
| `outer.py` | `_matmul_outer_kernel` 外积累加及 `_matmul_vector_kernel` 向量归约；`matmul_outer(a, b)` 为显式诊断入口 |
| `k1_dot.py` | 当前 dot 与外积对照，检查黄金 K=1 输入、补零和 tile 边界 |
| `k1_lowering.py` | 原始 K 专门化与当前动态 K 对照；用前置计算控制 SRAM 残留并导出 IR |
| `compile_dot.py` | 只编译最小/full dot，不启动 kernel |

迁移后 `FLAGOS_MATMUL_TRITON_ALGORITHM` 不再控制正式实现。需要外积
对照时，在项目根目录运行的 Python 中显式调用：

```python
from ops.matmul.probes.outer import matmul_outer

out = matmul_outer(a, b)
```

该入口支持原来的向量升降维、批次广播和空张量处理；M 或 N ≤ 2 时使用
向量归约，否则使用外积累加。它仅用于前向诊断，没有 autograd 注册。

个人容器、确认物理 XPU 2 空闲后可运行：

```bash
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/probes/k1_dot.py --dtype bfloat16 --repeats 3
```

`k1_dot.py` 打印的 `outer_rel` 是独立外积路径的误差，dot 使用当前生产
kernel。要复现已修复的旧编译错误，使用 `k1_lowering.py --variant original`，
具体参数和独立缓存要求见 [根因报告](../reports/k1_root_cause.md)。
