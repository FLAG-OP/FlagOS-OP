# softmax-fullstack: 行归约 + autotune

[← 样例索引](../README.md)

## 定位

补齐两个此前缺失的算子类别:

1. **reduction 类**: softmax 是行归约（max + exp + sum + div），
   与已有的 pointwise（gelu/silu）和 GEMM（bmm）构成 Triton 教程三部曲
2. **autotune**: `@triton.autotune` 自动从 5 个 BLOCK_N 配置中搜索最优

## 三层

- **L0 kernel**: 流式三遍归约 softmax（max → exp+sum → normalize），
  支持 N > BLOCK_N；精度 15/15 + 哨兵 + 性能
- **L2 aten**: `Library("aten","IMPL").impl("_softmax",...)` 拦截 F.softmax
- **L4 应用层**: attention scores softmax（手写 mini-attention）

## 运行

```bash
python3 examples/softmax-fullstack/example.py
```

## 实测

```
精度: 15/15 组合 PASS
autotune: 自动搜索 5 个 BLOCK_N（key=N）
性能: Triton ~0.0x ms vs F.softmax
```

## kernel 设计要点

- **流式三遍**: Pass1 找行 max → Pass2 算 exp 和 → Pass3 归一化写回
- **数值稳定**: 减 max 后再 exp
- **autotune**: 按 N 维度搜索 BLOCK_N，num_warps 联动
