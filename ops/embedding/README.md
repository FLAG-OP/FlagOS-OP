# embedding-op: `aten::embedding` P800/Kunlunxin 实现

按 FlagOS-OP 算子模板开发，范围为需求中的**普通稠密 Embedding 查表**：

- forward: 任意 rank indices 的 row gather；
- dense backward；
- FP32 / FP16 / BF16；
- `padding_idx`；
- `scale_grad_by_freq`；
- 显式排除 `sparse=True`、EmbeddingBag、稀疏优化器扩展。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)` |
| 平台 | p800-kunlunxin / torch_xmlir |
| 路线 | A1 `AutogradCUDA` 拦截 |
| forward 生产实现 | XMLIR native `aten::index_select` row gather |
| dense backward | native `aten::embedding_backward` |
| `scale_grad_by_freq` | per-occurrence inverse-frequency scaling + dense backward |
| 验证 | kernel 19 forward + 6 backward；黄金 117/117；A1；应用层全绿 |
| 性能 | 大 shape 与 native embedding 基本持平（0.89-1.00x）；Triton gather 慢 11-102x |

## 快速开始

```bash
python3 example.py p800-kunlunxin

python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/bench_perf.py --device cuda:1 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
```

## 实现策略

### Forward

P800 的 `aten::embedding` native 路径已经非常快。当前实现保留同名 A1
接口，但内部用：

```python
torch.ops.aten.index_select(weight, 0, flat_indices)
```

这避免 A1 注册后的 dispatcher 递归，同时复用 XMLIR 的 native row-gather。

多 rank indices 先 flatten，再恢复：

```text
(*indices.shape,) -> output.view(*indices.shape, embedding_dim)
```

### Backward

`scale_grad_by_freq=False`：

```text
aten::embedding_backward
```

`scale_grad_by_freq=True` 时 XPU native 报：

```text
Check scale_grad_by_freq == false failed
```

因此实现为：

1. 统计非 padding index 频率；
2. 对每个 occurrence 的 grad 乘 `1 / freq[index]`；
3. 再交给 dense backward 求和；
4. 保持 padding 行梯度为 0。

该路径与 CPU `aten::embedding_backward` 语义对齐。

## Triton 实验

`kernel/triton_level.py` 保留一个正确的固定块 Triton gather 探针，但不是
生产路径：

| shape | native | ours/index_select | Triton |
|---|---:|---:|---:|
| 1k×D128 | 0.0358ms | 0.0680ms | 0.2403ms |
| 16k×D128 | 0.0494ms | 0.0514ms | 2.2206ms |
| 131k×D128 | 0.1530ms | 0.1665ms | 16.9373ms |
| 16k×D512 | 0.0548ms | 0.0616ms | 8.7184ms |

结论：XMLIR native row-gather 是当前正确交付选择；Python/Triton 侧重写
不能带来收益。

## 边界

- `sparse=True` 显式 `NotImplementedError`；
- 不覆盖 EmbeddingBag；
- 不覆盖稀疏 / fused optimizer；
- weight 仅支持 FP32 / FP16 / BF16；
- indices 支持 int64 / int32；
- CPU profile 仅用于 reference，生产 backend 绑定 XMLIR/CUDA。
