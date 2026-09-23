# embedding 开发报告

## 1. 需求收窄

原始需求只要求普通 `aten.embedding` 查表，并明确不扩大到 EmbeddingBag、
稀疏反向或稀疏优化器。因此交付范围定义为：

- dense forward；
- dense backward；
- FP32/FP16/BF16；
- padding_idx；
- scale_grad_by_freq；
- explicit sparse rejection。

## 2. 平台探索

### Native

`aten::embedding` forward 在 P800 上非常快，native dense backward 也正确。
但：

```text
scale_grad_by_freq=true
```

触发 XPU native 未实现错误。

### FlagGems Triton

FlagGems `_kunlunxin` embedding kernel 每行启动一个 program。强制读回输出
后，16k×D128 约 3.7ms，而 native 约 0.05ms。

### 自写 Triton gather

实现过：

- row-per-program gather；
- 2D tile gather；
- flattened element gather。

正确性均可，但 16k×D128 约 2.2-3.5ms，不能与 native row-gather 竞争。

## 3. 生产方案

Forward：

```text
aten::index_select(weight, 0, flat_indices)
```

选择 index_select 的原因：

1. 复用 XMLIR native 高性能 gather；
2. 不经过 `aten::embedding` dispatcher，避免 A1 递归；
3. 与 native embedding 输出 bitwise 一致。

Backward：

```text
dense -> aten::embedding_backward
scale -> inverse-frequency scaling + dense backward
```

## 4. A1

`AutogradCUDA` 注册可以拦截：

- `torch.nn.functional.embedding`；
- `torch.embedding`；
- `nn.Embedding.forward`。

注册 wrapper 补齐 dispatcher 省略的默认参数，并用 `autograd.Function`
保持反向。

## 5. 测试设计

- reference 与官方 CPU embedding 交叉互验；
- 117 组黄金；
- forward rank/dtype/dim/padding/empty/int32；
- backward duplicate/padding/frequency；
- A1 direct bitwise；
- 应用层 `nn.Embedding` + MLP 前向、反向、贪心解码。

## 6. 结论

普通 embedding 是典型 gather-bound operator。P800 XMLIR native row-gather
已经足够强，当前自研 Triton 层无法带来收益。交付重点是：

1. 同名 A1 接口；
2. 任意 index rank；
3. 完整 dtype/backward 语义；
4. native 缺失的 inverse-frequency fallback；
5. 可复现测试与性能证据。
