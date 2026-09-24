# 算子开发报告: embedding

## 1. 环境配置

### 1.1 硬件

| 项 | 值 |
|---|---|
| 设备 | Kunlunxin P800 |
| 测试卡 | `cuda:1`（`CUDA_VISIBLE_DEVICES=1,2`） |

### 1.2 软件栈

| 项 | 值 |
|---|---|
| Python | 3.10.18 |
| Torch | 2.9.0+cu129 |
| torch_xmlir | XMLIR--bc1b1dc6f-dev+2026032411 |
| Triton | 3.0.0+03c4c9be |
| FlagGems | 4.2.1rc0 |

### 1.3 设备 profile 与关键环境变量

使用仓库 `configs/devices/p800-kunlunxin.yaml`。算子本地 profile 位于
`../_profile.py`，稳定设备为 `cuda:1`。

### 1.4 环境特殊性说明

`cuda:0` 曾出现厂商 SDPA 挂起；embedding 回归固定使用 `cuda:1`。XMLIR
异步计时必须读取输出元素，否则会只测 launch/submission。

## 2. 算子定义

### 2.1 语义

```text
out[*indices.shape, D] = weight[indices[i...], :]
```

`padding_idx` 不改变 forward；backward 时该行梯度为 0。
`scale_grad_by_freq=True` 时，每个 occurrence 的梯度先除以该非 padding
index 的出现次数，再按行求和。`sparse=True` forward 不改变查表结果；
sparse backward / sparse weight grad 不在本期范围。

### 2.2 PyTorch 参考实现

参考实现位于 `../reference.py`。forward 使用独立
`weight.index_select(...)`；backward 用 fp32 稠密累加实现语义参考。

### 2.3 接口签名

```python
embedding(
    weight,
    indices,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
) -> Tensor
```

### 2.4 数值规格

- FP32 / FP16 / BF16；
- int64 / int32 indices；
- forward 与 CPU 官方实现 bitwise equal；
- dense / inverse-frequency backward 与 CPU 语义参考误差为 0。

## 3. 实现说明

### 3.1 路线选择理由

P800 native row-gather 显著快于当前 Triton gather。为避免 A1 递归，
forward 内部使用等价的 `aten::index_select`；dense backward 使用
`aten::embedding_backward`。

### 3.2 kernel 实现要点

```text
forward:
  flatten indices
  aten::index_select(weight, 0, flat_indices)
  view(*indices.shape, D)

backward dense:
  aten::embedding_backward

backward scale_grad_by_freq:
  count non-padding indices
  grad *= 1/count[index]
  aten::embedding_backward(scale_grad_by_freq=False)
```

Triton gather 保留在 `kernel/triton_level.py` 作为探针；硬件级在
`kernel/hardware_level/README.md` 显式置空。

### 3.3 注册与分发

A1 注册点为 `AutogradCUDA`。wrapper 补齐 dispatcher 省略的默认参数，
并用 `autograd.Function` 保持 forward/backward。

## 4. 验证结果

### 4.1 kernel 层明细

- forward 20/20，max error 0；
- backward 6/6，max error 0；
- deterministic + input-sensitive 哨兵通过；
- sparse forward bitwise equal；
- sparse backward / invalid padding 显式拒绝。

### 4.2 framework 层明细

- 先启动 FlagOS/FlagGems：`flag_gems.only_enable(include=["gelu"])`；
- surrounding GELU 走 FlagGems，embedding 由 A1 backend 接管；
- A1 interception count 5；
- registered path 与 direct path bitwise equal；
- dense / inverse-frequency backward误差 0；
- `nn.Embedding` + MLP 应用层 logits 与梯度 diff 0，贪心续写一致率 1.00。

## 5. 性能

详见 [performance.md](performance.md)。生产 forward 在 16k 以上与 native
embedding 持平；Triton gather 慢约 3.5-102x。仓库 perf gate 已注册并入库
基线：`FAIL 0 · WARN 0 · NEW 0`。

## 6. 已知问题与风险

| 风险 | 处理 |
|---|---|
| `scale_grad_by_freq=True` fallback 产生中间 scaled gradient | 已量化，dense 1.33ms → scale 1.96ms |
| sparse backward 不覆盖 | forward 接受，backward 显式拒绝 |
| Triton gather 慢 | 仅保留平台探针，不接生产 |
| `cuda:0` 稳定性风险 | 回归固定 `cuda:1` |

## 7. 结论与后续

### 结论

普通 embedding 是 gather-bound operator。P800 XMLIR native row-gather
已经是当前最优路径；本交付补齐同名 A1 接口、任意 rank、dtype/backward
语义、native 缺失的 inverse-frequency fallback，以及完整测试与性能证据。

### 后续工作

如需 EmbeddingBag、sparse gradient 或 sparse/fused optimizer，应重新开
需求并设计稀疏存储、去重和更新链路，不应在当前 dense backend 上直接扩展。

## 附录 A: 复现命令

```bash
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
python3 script/bench_dispatch.py --device cuda:1 \
  --json-out reports/dispatch_p800-kunlunxin.json
```

## 附录 B: 相关产物

| 产物 | 位置 |
|---|---|
| operator perf JSON | `perf_fp16_p800-kunlunxin.json` |
| dispatch overhead JSON | `dispatch_p800-kunlunxin.json` |
| perf gate 基线 | `../../../perf/baselines/p800-kunlunxin.json` |
| 黄金规格 | `../goldendata/inputs_spec.yaml` |
