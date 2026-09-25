# embedding-op: `aten::embedding` P800/Kunlunxin + Cambricon MLU 实现

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
| 平台 | **2 个**：p800-kunlunxin / torch_xmlir；cambricon（MLU590）/ torch_mlu |
| 路线 | A1 拦截：P800 = `AutogradCUDA`，MLU = `AutogradPrivateUse1` |
| forward 生产实现 | 两平台均委托 native `aten::index_select` row gather |
| dense backward | native `aten::embedding_backward`（MLU 原生完整；XPU 缺分支见下） |
| `scale_grad_by_freq` | P800：per-occurrence inverse-frequency 补偿；MLU：原生实现直接委托 |
| 验证 | 两平台 kernel 20 forward + 6 backward；黄金 174/174 逐位；A1；应用层全绿 |
| 性能 | P800 大 shape 0.89-1.00x native；MLU 0.65-1.12x native；Triton gather 慢一个量级 |

## 快速开始

P800/Kunlunxin：

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

Cambricon MLU（`MLU_VISIBLE_DEVICES=7`，测试卡 `mlu:0`）：

```bash
python3 example.py cambricon

python3 script/check_accuracy.py --impl cambricon --device mlu:0
python3 script/bench_perf.py --device mlu:0 --dtype float16 \
  --json-out reports/perf_fp16_cambricon.json
python3 scripts/perf_run.py --device cambricon --pattern ops.embedding
python3 scripts/perf_compare.py --device cambricon
python3 script/bench_dispatch.py --device mlu:0 \
  --json-out reports/dispatch_cambricon.json
```

平台接入细节见 [reports/cambricon.md](reports/cambricon.md)。

## 实现策略

### Forward

P800 的 `aten::embedding` native 路径已经非常快。当前实现保留同名 A1
接口，但内部用：

```python
torch.ops.aten.index_select(weight, 0, flat_indices)
```

这避免 A1 注册后的 dispatcher 递归，同时复用 XMLIR 的 native row-gather。

Cambricon MLU 侧同样委托 `aten::index_select`，与 CPU 参考逐位一致
（`kernel/cambricon.py`）。两平台由 [kernel/platform.py](kernel/platform.py)
按 `tensor.device.type` 路由，测试、bench 与 A1 注册共用同一条调用路径。

多 rank indices 先 flatten，再恢复：

```text
(*indices.shape,) -> output.view(*indices.shape, embedding_dim)
```

### Backward

`scale_grad_by_freq=False`：

```text
aten::embedding_backward
```

**Cambricon MLU：原生 `aten::embedding_backward` 完整**——`padding_idx` 与
`scale_grad_by_freq=True` 都已实现且与 CPU 参考 6/6 组合 `err=0`，因此直接
委托、无任何补偿代码（与 P800 的 XPU 方案形成对照）。

**P800/Kunlunxin** 上 `scale_grad_by_freq=True` 时 XPU native 报：

```text
Check scale_grad_by_freq == false failed
```

因此实现为：

1. 统计非 padding index 频率；
2. 对每个 occurrence 的 grad 乘 `1 / freq[index]`；
3. 再交给 dense backward 求和；
4. 保持 padding 行梯度为 0。

该路径与 CPU `aten::embedding_backward` 语义对齐（仅 P800/XPU 需要；
MLU 直接走原生分支）。

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

MLU 上同一探针更不具竞争力（fp16，[perf_fp16_cambricon.json](reports/perf_fp16_cambricon.json)）：

| shape | native | ours/index_select | Triton |
|---|---:|---:|---:|
| 1k×D128 | 0.0445ms | 0.0681ms | 0.2995ms |
| 16k×D128 | 0.0451ms | 0.0612ms | 2.3029ms |
| 131k×D128 | 0.0662ms | 0.0819ms | 失败（`grid=65536 > 65535`） |
| 16k×D512 | 0.0617ms | 0.0550ms | 10.5459ms |

即比 native 慢 6.7-171x，且 131k indices 会撞 Triton grid 上限直接 launch
失败——Triton gather 在两平台都只作精度对照。

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
- CPU profile 仅用于 reference，生产 backend 绑定 XMLIR/CUDA（P800）与
  torch_mlu（cambricon），由 [kernel/platform.py](kernel/platform.py) 按设备路由。
