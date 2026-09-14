# softmax-fullstack: 行归约算子贯穿三层（样板同构组织）

本样例的目录结构与 [templates/operator](../../templates/operator/README.md)
完全一致——从样板开始开发的人切到本样例**零认知切换**；未实现的
级别置空并说明（[kernel/hardware_level](kernel/hardware_level/README.md)）。

```
softmax-fullstack/
├── REPORT.md + reports/     总报告 + 交付报告四件套（开发/测试/精度/性能，已填真实数据）
├── reference.py             语义参考
├── register.py              A1 aten 注册
├── kernel/                  torch ✅ · Triton ✅ · hardware ⬜置空
├── test/                    算子库层 / 框架层 / 应用层
├── goldendata/              规格含 #15 尾块回归形状
└── script/                  gen_golden / check_accuracy
```

[← 样例索引](../README.md)

## 定位

补齐两个此前缺失的算子类别:

1. **reduction 类**: softmax 是行归约（max + exp + sum + div），
   与已有的 pointwise（gelu/silu）和 GEMM（bmm）构成 Triton 教程三部曲
2. **尾块安全**: 非 BLOCK 整数倍的 N 自动 pad 到 2048 倍数（known-issues #15a）

## 三层

- **kernel 层**: 流式三遍归约 softmax（max → exp+sum → normalize），
  支持 N > BLOCK_N；精度 15/15 + 哨兵 + 性能
- **框架层 aten**: `Library("aten","IMPL").impl("_softmax",...)` 拦截 F.softmax
- **应用层**: attention scores softmax（手写 mini-attention）

## 运行

```bash
python3 examples/softmax-fullstack/example.py
```

## 实测

```
精度: 15/15 组合 PASS
配置: BLOCK_N=2048 固定（autotune 在本栈会选出非法 num_warps=5，#15b，已移除）
性能: Triton ~0.0x ms vs F.softmax
```

## kernel 设计要点

- **流式三遍**: Pass1 找行 max → Pass2 算 exp 和 → Pass3 归一化写回
- **数值稳定**: 减 max 后再 exp
- **固定 BLOCK_N=2048**: 本栈 autotuner 不可信（#15b）；整数倍 N 零 pad 开销
