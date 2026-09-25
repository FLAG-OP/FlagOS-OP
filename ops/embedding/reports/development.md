# 算子开发报告: embedding

## 1. 环境配置

### 1.1 硬件

| 项 | P800 | Cambricon MLU |
|---|---|---|
| 设备 | Kunlunxin P800 | MLU590-M9 |
| 测试卡 | `cuda:1`（`CUDA_VISIBLE_DEVICES=1,2`） | `mlu:0`（`MLU_VISIBLE_DEVICES=7`，driver 6.2.29） |

### 1.2 软件栈

| 项 | P800 | Cambricon MLU |
|---|---|---|
| Python | 3.10.18 | 3.12.13 |
| Torch | 2.9.0+cu129 | 2.11.0+cpu |
| 厂商插件 | torch_xmlir `XMLIR--bc1b1dc6f-dev+2026032411` | torch_mlu `1.33.1+torch2.11.0` |
| Triton | 3.0.0+03c4c9be | 3.4.0+mlu2.1.1 |
| FlagGems | 4.2.1rc0 | 5.3.5（`_cambricon`） |

### 1.3 设备 profile 与关键环境变量

P800 使用仓库 `configs/devices/p800-kunlunxin.yaml`，稳定设备为 `cuda:1`；
cambricon 使用 `configs/devices/cambricon.yaml`，测试卡 `mlu:0`。
算子本地 profile 位于 `../_profile.py`，别名 `mlu` / `mlu590` /
`cambricon-mlu590` 均归一到 `cambricon`。

### 1.4 环境特殊性说明

`cuda:0` 曾出现厂商 SDPA 挂起；embedding 回归固定使用 `cuda:1`。XMLIR
异步计时必须读取输出元素，否则会只测 launch/submission——MLU 同样保留
该完成守卫（`reshape(-1)[0].item()`），两平台计时口径一致。

cambricon 侧：`scripts/check_env.py --device cambricon` 报 6 项不符，是
`d5c7a3d` 写锁时的容器镜像与本机 torch2.11 栈的**既有漂移**（与本改动无关）；
`--device mlu590` 为 OK。

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

A1 注册点按平台：P800 = `AutogradCUDA`，cambricon = `AutogradPrivateUse1`
（两个 key 实测都能命中，但 `PrivateUse1` 会覆盖 functorch 的 batch rule，
且 Autograd key 才能携带自带 backward——与 `AutogradCUDA` 同构）。wrapper
补齐 dispatcher 省略的默认参数，并用 `autograd.Function` 保持
forward/backward。

### 3.4 多平台结构（cambricon 接入）

```text
kernel/
  p800_kunlunxin.py   P800 后端（字节未改）
  cambricon.py        MLU 后端：前向 aten::index_select
                             反向 aten::embedding_backward（原生完整）
  platform.py         按 tensor.device.type 路由的门面
                      + dispatch_key_for / synchronize
```

选择扁平结构而非子目录，理由与 `ops/sdpa/kernel/triton_level.py` 一致：
调用方只 import 一次，设备后端在调用时决定，测试/bench/A1 注册共用一条路径。

MLU 后端的实现选择（详见 [cambricon.md](cambricon.md)）：

- 前向 `aten::index_select` 与 CPU 逐位一致，直接委托；
- 反向原生 `embedding_backward` 覆盖 `padding_idx` 与
  `scale_grad_by_freq`（6/6 组合 err=0），**不需要** P800 的
  inverse-frequency 补偿；
- `sparse=True` backward 显式 `NotImplementedError`（torch_mlu 会返回
  COO 稀疏梯度而不是报错，必须自己拦）；
- 平台/profile/门禁命名统一 `cambricon`，别名归一。

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

### 4.3 cambricon 层明细

- forward 20/20、backward 6/6，max error 0（含 `scale_grad_by_freq` 原生分支）；
- 黄金 174/174，worst=0（同一套黄金、同一逐位判据）；
- A1 拦截 5 次，注册路径与直调路径逐位一致；
- 应用层拦截 8 次，logits / 两级梯度 diff 0，贪心一致率 1.00，
  FlagGems `gelu` 生效；
- `example.py cambricon` 三层 PASS；`pytest tests/unit -q` 20 passed。

## 5. 性能

详见 [performance.md](performance.md)。P800 生产 forward 在 16k 以上与 native
embedding 持平；Triton gather 慢约 3.5-102x。cambricon 侧 ours/native 落在
0.65-1.12x（唯一超过 native 的是 16k×D512 的 1.123x），Triton gather 慢
6.7-171x 且 131k indices 撞 grid 65535 上限。两平台 perf gate 均
`FAIL 0 · WARN 0 · NEW 0`；cambricon 基线新增 2 条用例（共 33 条）。

A1 dispatch（MLU，修复 bench 脚本 Library GC 缺陷后）：native 0.0450ms /
direct 0.0600ms / A1 0.1041ms，A1 − native ≈ +0.059ms；P800 同口径数字作废
待复测。

## 6. 已知问题与风险

| 风险 | 处理 |
|---|---|
| `scale_grad_by_freq=True` fallback 产生中间 scaled gradient | P800 已量化，dense 1.33ms → scale 1.96ms；MLU 走原生仅 +11% |
| sparse backward 不覆盖 | forward 接受，backward 显式拒绝（MLU 侧必须自己拦，torch_mlu 返回 COO） |
| Triton gather 慢 | 仅保留平台探针，不接生产（MLU 131k 撞 grid 上限） |
| `cuda:0` 稳定性风险 | 回归固定 `cuda:1` |
| `bench_dispatch.py` 曾漏持 `Library` | 已修（持有 `_a1_lib`）；P800 a1 数字作废待复测，MLU 已重测 |
| `check_env --device cambricon` 6 项不符 | 既有锁漂移，非本次改动引入 |

## 7. 结论与后续

### 结论

普通 embedding 是 gather-bound operator。P800 XMLIR native row-gather
已经是当前最优路径；本交付补齐同名 A1 接口、任意 rank、dtype/backward
语义、native 缺失的 inverse-frequency fallback，以及完整测试与性能证据。

第二平台 cambricon 复用同一套黄金与三层测试，验证了「同名接口 + 厂商原生
row-gather 委托」的结构可扩展性：MLU 原生 backward 完整，后端是零补偿的纯
委托；两平台唯一差异收敛在 `kernel/*.py` 一个文件与 A1 dispatch key。

### 后续工作

如需 EmbeddingBag、sparse gradient 或 sparse/fused optimizer，应重新开
需求并设计稀疏存储、去重和更新链路，不应在当前 dense backend 上直接扩展。
另需在有 `torch_xmlir` 的环境复跑 P800 的 `bench_dispatch.py`，补齐作废的
A1 dispatch 数字。

## 附录 A: 复现命令

```bash
# P800 / Kunlunxin
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
python3 script/bench_dispatch.py --device cuda:1 \
  --json-out reports/dispatch_p800-kunlunxin.json

# Cambricon MLU590
export MLU_VISIBLE_DEVICES=7
python3 example.py cambricon
python3 script/check_accuracy.py --impl cambricon --device mlu:0
python3 script/bench_perf.py --device mlu:0 --dtype float16 \
  --json-out reports/perf_fp16_cambricon.json
python3 scripts/perf_run.py --device cambricon --pattern ops.embedding
python3 scripts/perf_compare.py --device cambricon
python3 script/bench_dispatch.py --device mlu:0 \
  --json-out reports/dispatch_cambricon.json
```

## 附录 B: 相关产物

| 产物 | 位置 |
|---|---|
| operator perf JSON（P800） | `perf_fp16_p800-kunlunxin.json` |
| dispatch overhead JSON（P800） | `dispatch_p800-kunlunxin.json` |
| operator perf JSON（cambricon） | `perf_fp16_cambricon.json` |
| dispatch overhead JSON（cambricon） | `dispatch_cambricon.json` |
| cambricon 平台报告 | [cambricon.md](cambricon.md) |
| perf gate 基线 | `../../../perf/baselines/p800-kunlunxin.json`、`../../../perf/baselines/cambricon.json` |
| 黄金规格 | `../goldendata/inputs_spec.yaml` |
