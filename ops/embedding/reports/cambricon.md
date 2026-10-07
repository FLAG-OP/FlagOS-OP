# cambricon（Cambricon MLU590）平台接入报告

日期：2026-09-24（dispatch 与 perf 复测 2026-09-25）

本算子原为 P800/Kunlunxin 单平台实现，本次按仓库既有的「三层实现 + 黄金数据」
标准接入第二平台 Cambricon MLU。`kernel/p800_kunlunxin.py` 逐字节未改动。

## 1. 环境

| 项 | 值 |
|---|---|
| Torch | 2.11.0+cpu（torch_mlu 后端 1.33.1+torch2.11.0） |
| torch_mlu | 1.33.1+torch2.11.0 |
| Triton | 3.4.0（`+mlu2.1.1`） |
| FlagGems | 5.3.5（`_cambricon`，editable main） |
| CNNL / CNRT | 2.2.14 / 7.7.2 |
| 设备 | MLU590-M9，driver 6.2.29，`MLU_VISIBLE_DEVICES=7`，测试卡 `mlu:0` |
| Python / numpy | 3.12.13 / 1.26.4 |

环境核对：

```bash
python3 scripts/check_env.py --device cambricon
python3 scripts/check_env.py --device mlu590
```

> `--device mlu590` → **OK**；`--device cambricon` → **6 项不符**（锁定期望
> torch 2.12.1 / torch_mlu 1.34.1 / triton +mlu2.1.3 / flag_gems 5.0.2 /
> numpy 2.2.6 / py 3.12.3）。这是 `d5c7a3d` 写锁时的容器镜像与本机实测栈的
> **既有漂移**（本机是 torch2.11 栈），与本次 embedding 改动无关，未在本 PR 处理。

## 2. 实现选择

### 2.1 路径：原生委托，不做厂商补偿

| 候选路径 | 实测结论 |
|---|---|
| 手写 inverse-frequency 补偿（沿用 P800 XPU 方案） | **不需要**——MLU 原生 `embedding_backward` 完整 |
| `reference.embedding_backward_reference` | 仅 CPU：循环内直接建 CPU tensor，MLU 上不可用 |
| `aten::index_select` 作前向 | 与 CPU 参考 **逐位一致**，直接采用 |
| `sparse=True` 反向 | torch_mlu 会返回 COO 稀疏梯度而非报错 → **显式 `raise NotImplementedError` 拦下** |

最终路径（与 P800 的「原生 gather + 自实现反向补偿」形成对照）：

```
业务 F.embedding / nn.Embedding
  → A1 (AutogradPrivateUse1) → kernel/platform.py 按 device 路由
      ├─ mlu   → kernel/cambricon.py   前向 aten::index_select
      │                    反向 aten::embedding_backward（原生，含 padding_idx +
      │                    scale_grad_by_freq，与 CPU 参考 6/6 组合 err=0）
      └─ cuda  → kernel/p800_kunlunxin.py（未改动）
```

- 前向 `sparse=True` 只请求稀疏**梯度**，不改查找语义，故前向接受、反向拦截
  （`kernel/cambricon.py::_validate_forward` 注释）。
- `F.embedding(padding_idx=-1)` 在进 aten 之前先归一化为
  `weight.size(0)+padding_idx`（`torch/nn/functional.py::embedding`，本机
  第 2552 行 `padding_idx = weight.size(0) + padding_idx`），所以 A1 收到的
  是归一化后的值，与 CPU 逐位相同；facade 直调 `-1` 不会误清零任何一行——
  已按语义核对，非缺陷。

### 2.2 结构：保持扁平，不引入子目录

| 文件 | 作用 |
|---|---|
| [`kernel/cambricon.py`](../kernel/cambricon.py) | MLU 后端（校验 + 原生委托 + 热路径） |
| [`kernel/platform.py`](../kernel/platform.py) | 按 `tensor.device.type` 路由的门面，附 `dispatch_key_for` / `synchronize` |
| [`kernel/p800_kunlunxin.py`](../kernel/p800_kunlunxin.py) | 原实现，字节未动 |

门面复用 `ops/sdpa/kernel/triton_level.py` 的写法：调用方只 import 一次，
设备后端在调用时由 `weight.device.type` 决定，测试/bench/A1 注册全部走同一条路径。

## 3. 平台命名

平台 / profile / 性能门禁一律用 **`cambricon`**，别名 `mlu`、`mlu590`、
`cambricon-mlu590` 在 `_profile.load_profile` 归一化到 `cambricon`。
理由与其余 13 个已接入 MLU 的算子一致：与 `perf/baselines/cambricon.json`、
`configs/devices/cambricon.yaml`、flag_gems `DeviceDetector().vendor_name` 同名。

## 4. A1 注册与 dispatch key

| 项 | 结论 |
|---|---|
| dispatch key | **`AutogradPrivateUse1`** |
| 两个 key 都能命中吗 | 能。`AutogradPrivateUse1` 与 `PrivateUse1` 实测都能拦截到调用 |
| 为什么不用 `PrivateUse1` | 它会覆盖 functorch 的 batch rule（`functorch/BatchRulesModules.cpp` 注册的 kernel 被顶掉，运行时报 "Overriding a previously registered kernel…" warning） |
| 为什么与 sdpa 不同 | sdpa 是复合算子可走 `PrivateUse1`；embedding 是带自带 backward 的 A1 交付，必须注册到 **Autograd** key 才能同时覆盖 grad/no_grad——与 p800 的 `AutogradCUDA` 同构 |
| 与 `configs/devices/cambricon.yaml` 的差异 | yaml 写 `dispatch_key: PrivateUse1`（设备后端通用值）；embedding profile 写 `AutogradPrivateUse1`，`_profile.py` 内有注释说明差异来源 |
| 拦截计数 | op 层 5 次、framework 层 8 次（见 §5） |

注册与直调路径逐位一致：`op_level.py` 的 `bitwise_vs_direct=True`。

## 5. 验证结果

```bash
python3 test/kernel_level.py cambricon
python3 script/check_accuracy.py --impl cambricon --device mlu:0
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 example.py cambricon
cd ../.. && python3 -m pytest tests/unit -q
```

| 层 | 结果 |
|---|---|
| kernel 层 | ✅ 前向 **20/20**（1d/2d-padding/3d/d80-tail/d512/empty/int32 × f32/f16/bf16），反向 **6/6**（3 dtype × `scale_grad_by_freq` 0/1）`err=0.000e+00`，哨兵通过，守卫 2 项通过；16k×D128 采样 0.063ms |
| 黄金数据 | ✅ **174/174 PASS，worst=0.000e+00**（容差 0.0 逐位；cambricon/triton/native/torch 四实现同判据） |
| op 层 | ✅ A1 拦截 5 次，注册=直调（逐位），dense 梯度 err=0，`scale_grad_by_freq` 梯度 err=0，`sparse` 前向通过、反向被拒 |
| 应用层 | ✅ mini-decoder（nn.Embedding + 2 层 MLP，bf16，4 prompts × 48 tokens）拦截 8 次，logits diff=0.000e+00，greedy 一致率 1.00，embedding/首层 linear 梯度 diff=0；`flag_gems.only_enable(['gelu'])` 生效 |
| 单测 | ✅ `pytest tests/unit -q` **20 passed** |
| 三级冒烟 | ✅ `python3 example.py cambricon` → kernel / op / application 三层 PASS |

## 6. 性能

数据：[perf_fp16_cambricon.json](perf_fp16_cambricon.json)（warmup=20，iters=100，
每次读回一个输出元素强制 kernel 完成；speedup = native/ours，**>1 表示 ours 更快**）。

### Forward（fp16）

| shape | ours | native | Triton gather | ours/native |
|---|---:|---:|---:|---:|
| 1k × D128 | 0.0681ms | 0.0445ms | 0.2995ms | 0.654x |
| 16k × D128 | 0.0612ms | 0.0451ms | 2.3029ms | 0.736x |
| 131k × D128 | 0.0819ms | 0.0662ms | 失败（grid 65536 > 65535） | 0.809x |
| 16k × D512 | 0.0550ms | 0.0617ms | 10.5459ms | **1.123x** |
| vocab128k 16k × D128 | 0.0525ms | 0.0404ms | 2.3306ms | 0.769x |

### Backward（16k × D128，fp16）

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.2651ms | 0.2497ms | 0.942x |
| `scale_grad_by_freq=True` | 0.2944ms | 0.2796ms | 0.950x |

> MLU 上 `scale_grad_by_freq` 的额外代价约 **+11%**（0.2651→0.2944ms），
> 远小于 P800 上手写 inverse-frequency 的 +47%——原生实现自带该分支。

### 仓库 perf gate

```bash
python3 scripts/perf_run.py --device cambricon --pattern ops.embedding
python3 scripts/perf_compare.py --device cambricon
```

| case | latency | 带宽估算 |
|---|---:|---:|
| `ops.embedding.cambricon.forward` | 0.060ms | 142.8 GB/s |
| `ops.embedding.native.forward` | 0.045ms | 187.9 GB/s |

```text
FAIL 0 · WARN 0 · NEW 0 · 结论 OK
```

`perf/baselines/cambricon.json` 现有 **33** 条 case（原 31 + 本算子 2 条）；
P800 侧 `ops.embedding.p800.forward` 保留不动。

## 7. A1 dispatch 开销（含一个 bench 缺陷修正）

数据：[dispatch_cambricon.json](dispatch_cambricon.json)（native / direct / a1
各在独立子进程测，4096×D128、16384 indices、fp16、median/p20/p80）。

| mode | median | p20 | p80 |
|---|---:|---:|---:|
| native `aten::embedding` | 0.0450ms | 0.0448ms | 0.0454ms |
| direct backend（facade 直调） | 0.0600ms | 0.0596ms | 0.0608ms |
| A1 `F.embedding` | 0.1041ms | 0.1035ms | 0.1054ms |

A1 − native ≈ **+0.059ms**，A1 − direct ≈ **+0.044ms**。
2026-09-25 复测 3 轮，A1 − native = 0.059 / 0.060 / 0.061ms，结论稳定。

### 缺陷：`torch.library.Library` 析构即反注册

`register_a1(...)` 返回的 `Library` 若不持有引用，**Python GC 会在语句结束时
析构它，随即把刚注册的 kernel 反注册掉**。原 `bench_dispatch.py` 的 a1 分支是
裸调用：

```python
register_a1(dispatch_key_for(dev))   # 旧写法：lib 立即被回收
```

于是 a1 子进程实际测的是**未注册状态**（`F.embedding` 落回原生路径）。修正为：

```python
_a1_lib = register_a1(dispatch_key_for(dev))   # 持有引用
assert _a1_lib is not None
```

- 全仓扫描确认这是唯一裸调用点（sdpa 的调用点与两个测试文件都持有 `lib`）。
- **精度结论不受影响**：三层测试里 `lib` 一直被持有，拦截计数 5/8 均 > 0。
- **P800 侧 a1 数字作废**：历史 `+0.0241ms`、"A1 与 direct 同档/略快" 的结论
  来自未注册状态，需复测；本机无 `torch_xmlir`（`ModuleNotFoundError`），
  **无法重测**，`REPORT.md` / `reports/performance.md` / `reports/test-report.md` /
  `CHANGELOG.md` 中相应行已标注「待复测」。native / direct 两行不受影响，仍然有效。
- MLU 侧数字即上表（修正后重测）。

## 8. 关键平台发现

1. **MLU 原生 `embedding_backward` 是完整的**：`padding_idx` 与
   `scale_grad_by_freq` 都实现且与 CPU 参考 6/6 组合逐位一致——与 XPU 缺
   `scale_grad_by_freq` 必须自补偿形成直接对照，因此 MLU 后端无任何厂商补偿代码。
2. **前向走 `aten::index_select` 逐位等于 CPU**，是本栈最快的一方原生入口。
3. **`sparse=True` 反向不报错而是返回 COO**：若不拦截，用户会拿到 torch_mlu 的
   稀疏梯度；必须显式 `NotImplementedError`（前向仍接受，语义不变）。
4. **A1 key 必须用 `AutogradPrivateUse1`**：`PrivateUse1` 会顶掉 functorch 的
   batch rule（`BatchRulesModules.cpp`）；Autograd key 同时与 p800 `AutogradCUDA`
   构成同构模式。
5. **`torch.library.Library` 不持引用会被 GC 反注册**，任何「注册后测延迟」的
   脚本都必须持有返回的 lib，否则测的是未注册状态。
6. **131k indices 触发 Triton grid 上限**：`grid=65536 > 65535` 直接失败，
   Triton gather 只能当精度对照，不能当实现。
7. `F.embedding(padding_idx=-1)` 的归一化发生在进 aten 之前
   （`torch/nn/functional.py::embedding` 第 2552 行），A1 收到的已是
   `num_weights-1`，与 CPU 逐位一致。
8. `scale_grad_by_freq` 在 MLU 上代价 +11%（P800 手写补偿 +47%）。

## 9. 复现命令

```bash
cd ops/embedding
export MLU_VISIBLE_DEVICES=7

python3 example.py cambricon                 # 三级冒烟
python3 test/kernel_level.py cambricon
python3 script/check_accuracy.py --impl cambricon --device mlu:0
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/bench_perf.py --device mlu:0 --dtype float16 \
  --json-out reports/perf_fp16_cambricon.json
python3 script/bench_dispatch.py --device mlu:0 \
  --json-out reports/dispatch_cambricon.json

cd ../.. && python3 -m pytest tests/unit -q
python3 scripts/perf_run.py --device cambricon --pattern ops.embedding
python3 scripts/perf_compare.py --device cambricon
```
