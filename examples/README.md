# 样例索引（路线 × 测试层级 = 6 格）

每个样例目录 = 一个矩阵格的完整可运行演示。

| 样例 | 内容 | 运行 |
|---|---|---|
| [a1-op](a1-op/) | 从零写 Triton relu → aten 注册 → 拦截/精度/性能 | `python3 examples/a1-op/example.py` |
| [a1-framework](a1-framework/) | aten::silu 恒等计数注入真实 vLLM 前向 | `python3 examples/a1-framework/example.py` |
| [a2-op](a2-op/) | Triton gelu_and_mul → dispatch 双后端 → 三段式 | `python3 examples/a2-op/example.py` |
| [a2-framework](a2-framework/) | Triton silu_and_mul 以 vendor 身份注入真实推理 | `python3 examples/a2-framework/example.py` |
| [b-op](b-op/) | 自定义 vendor backend 注册/选择/计数 | `python3 examples/b-op/example.py` |
| [b-framework](b-framework/) | audit vendor 拦截真实 vLLM + 黄金回归 | `python3 examples/b-framework/example.py` |

样例与正式测试（`run.py`）复用同一套基础设施（harness / golden / 设备 profile）；
算子层样例完全自包含（不依赖 routes/），可直接复制为开发起点。
