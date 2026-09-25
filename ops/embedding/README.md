# embedding-op: `aten::embedding` P800/Kunlunxin 实现

按 FlagOS-OP 算子模板开发，范围为需求中的**普通稠密 Embedding 查表**：

- forward: 任意 rank indices 的 row gather；
- dense backward；
- FP32 / FP16 / BF16；
- `padding_idx`；
- `scale_grad_by_freq`；
- `sparse=True` forward 按普通查表接受；不扩展稀疏反向。
- 显式排除 EmbeddingBag、稀疏反向、稀疏优化器扩展。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)` |
| 平台 | p800-kunlunxin / torch_xmlir |
| 路线 | A1 `AutogradCUDA` 拦截 |
| forward 生产实现 | XMLIR native `aten::index_select` row gather |
| dense backward | native `aten::embedding_backward` |
| `scale_grad_by_freq` | per-occurrence inverse-frequency scaling + dense backward |
| 验证 | kernel 20 forward + 6 backward；黄金 174/174；A1；应用层全绿 |
| 性能 | 大 shape 与 native embedding 基本持平（0.89-1.00x）；Triton gather 慢 11-102x |

## 快速开始

```bash
python3 example.py p800-kunlunxin

python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/bench_perf.py --device cuda:1 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_compare.py --device p800-kunlunxin
python3 script/bench_dispatch.py --device cuda:1 \
  --json-out reports/dispatch_p800-kunlunxin.json
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

### 与 FLAG-OP/gatherFIX 的关系

组织内已有 [FLAG-OP/gatherFIX](https://github.com/FLAG-OP/gatherFIX)。
它修复的是 **Ascend `torch.gather` 非连续 index 越界**（FlagGems issue
#5746）：原 kernel 将非连续 index 当作线性平铺读取，修复引入
stride-aware kernel。

为了避免误用结论，本 PR 将该 stride-aware 算法特化成 P800 embedding
row lookup 并实测（`script/probe_gatherfix.py`）：

| shape | native embedding | ours/index_select | gatherFIX 特化 |
|---|---:|---:|---:|
| 1k×D128 | 0.0386ms | 0.0392ms | 0.4213ms |
| 16k×D128 | 0.0705ms | 0.0738ms | 3.2452ms |
| 131k×D128 | 0.2696ms | 0.3595ms | 22.6192ms |
| 16k×D512 | 0.0636ms | 0.0763ms | 11.3235ms |

结果：正确性 0 error，但该修复是 Ascend correctness fix，不表示 P800
embedding / `index_select` 的 Triton 性能已达 native。生产路径继续使用
XMLIR native row gather。

该结论不等价于：

```text
aten::embedding(dim=0 row lookup)
aten::index_select
P800/Kunlunxin Triton gather 性能已达到 native row-gather
```

本页数字来自当前锁定镜像与组织 gatherFIX 算法的 P800 特化探针；更新版
FlagGems 仍应按环境升级 runbook 重测。

## FlagOS 框架测试口径

框架/应用验证运行在 FlagOS 算子栈内：

```python
import flag_gems
flag_gems.only_enable(include=["gelu"])  # 消费方 surrounding op 走 FlagGems
```

当前锁定镜像上全量 `flag_gems.enable()` 在该 consumer 上存在非确定性，
因此框架层选择稳定且真实被模型调用的 `gelu` 作为 FlagOS/FlagGems 代表
路径；`aten::embedding` 由本目录 A1 注册接管。

## 边界

- `sparse=True` forward 不改变输出，按 dense lookup 返回；
- `sparse=True` backward / sparse weight grad 显式 `NotImplementedError`；
- 不覆盖 EmbeddingBag；
- 不覆盖稀疏 / fused optimizer；
- weight 仅支持 FP32 / FP16 / BF16；
- indices 支持 int64 / int32；
- CPU profile 仅用于 reference，生产 backend 绑定 XMLIR/CUDA。
