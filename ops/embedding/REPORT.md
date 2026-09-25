# 算子总体报告: embedding

| 项 | 值 |
|---|---|
| 算子 | `aten::embedding` |
| 范围 | 普通查表 forward + dense backward；`sparse=True` forward 可用，稀疏 backward 不覆盖 |
| 硬件平台 | **2 个**：Kunlunxin P800（profile `p800-kunlunxin`，`torch_xmlir`）；Cambricon MLU590（profile `cambricon`，`torch_mlu`） |
| 路线 | A1：P800 = `AutogradCUDA`，MLU = `AutogradPrivateUse1` |
| 日期 / 状态 | 2026-09-24 / 两平台三层与黄金全绿 |

## 实现矩阵（三级）

| 平台 / 级别 | 文件 | 状态 | 说明 |
|---|---|---|---|
| P800 厂商委托 | [kernel/p800_kunlunxin.py](kernel/p800_kunlunxin.py) | ✅ | forward 走 native `index_select`；dense backward 走 native `embedding_backward`；XPU 缺 `scale_grad_by_freq` → 自实现 inverse-frequency 补偿 |
| Cambricon MLU 委托 | [kernel/cambricon.py](kernel/cambricon.py) | ✅ | forward 走 native `index_select`；backward 走 native `embedding_backward`（`padding_idx`/`scale_grad_by_freq` 完整，无需补偿）；`sparse=True` backward 显式拒绝 |
| 多平台门面 | [kernel/platform.py](kernel/platform.py) | ✅ | 按 `tensor.device.type` 路由后端，附 `dispatch_key_for` / `synchronize`；测试与 A1 注册共用 |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ 探针 | 正确但比 native gather 慢，未接入生产 |
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | 独立 index-select 参考/对照 |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 | native row-gather 已达到生产水位，暂无重写收益 |

## 验证矩阵（三层，P800）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py p800-kunlunxin` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6；重复 index、padding、`scale_grad_by_freq` 全部 0 error |
| 黄金 | `script/check_accuracy.py --impl p800` | 174/174，worst=0 |
| A1 op | `test/op_level.py` | 拦截 5 次；hooked=direct 逐位；dense/scale/sparse 边界全绿 |
| 应用层 | `test/framework_level.py`（FlagGems enabled） | `nn.Embedding` + MLP；logits/梯度 0 diff；贪心一致率 1.00 |
| 性能 | `script/bench_perf.py` | 大 shape native parity；Triton 显著慢 |
| dispatch | `script/bench_dispatch.py` | native / direct 行有效；**A1 行作废待复测**（见遗留 4） |

## 验证矩阵（三层，cambricon / MLU590）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py cambricon` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6 全部 `err=0`（含 `scale_grad_by_freq`，原生实现）；守卫 2 项通过 |
| 黄金 | `script/check_accuracy.py --impl cambricon --device mlu:0` | 174/174，worst=0（逐位） |
| A1 op | `test/op_level.py cambricon` | 拦截 5 次；hooked=direct 逐位；dense/scale grad 0 error；sparse fwd 接受 / bwd 拒绝 |
| 应用层 | `test/framework_level.py cambricon` | 拦截 8 次；logits 0 diff；贪心一致率 1.00；FlagGems gelu 生效 |
| 性能 | `script/bench_perf.py --device mlu:0` | 见下表；gate `FAIL 0 · WARN 0 · NEW 0` |
| dispatch | `script/bench_dispatch.py --device mlu:0` | A1 − native ≈ **+0.059ms**（3 轮 0.059/0.060/0.061） |
| 单测 | `pytest tests/unit -q` | 20 passed |

## 关键数字

P800 forward（fp16）：

| shape | ours | native | Triton | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0680ms | 0.0358ms | 0.2403ms | 0.526x |
| 16k×D128 | 0.0514ms | 0.0494ms | 2.2206ms | 0.963x |
| 131k×D128 | 0.1665ms | 0.1530ms | 16.9373ms | 0.919x |
| 16k×D512 | 0.0616ms | 0.0548ms | 8.7184ms | 0.889x |
| vocab128k 16k×D128 | 0.0514ms | 0.0515ms | 2.2458ms | 1.001x |

P800 backward 16k×D128：

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 1.3306ms | 1.2993ms | 0.976x |
| scale_grad_by_freq | 1.9562ms | XPU native 不支持 | — |

cambricon forward（fp16，[perf_fp16_cambricon.json](reports/perf_fp16_cambricon.json)）：

| shape | ours | native | Triton | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0681ms | 0.0445ms | 0.2995ms | 0.654x |
| 16k×D128 | 0.0612ms | 0.0451ms | 2.3029ms | 0.736x |
| 131k×D128 | 0.0819ms | 0.0662ms | 失败（grid 65536>65535） | 0.809x |
| 16k×D512 | 0.0550ms | 0.0617ms | 10.5459ms | **1.123x** |
| vocab128k 16k×D128 | 0.0525ms | 0.0404ms | 2.3306ms | 0.769x |

cambricon backward 16k×D128：

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.2651ms | 0.2497ms | 0.942x |
| scale_grad_by_freq | 0.2944ms | 0.2796ms | 0.950x |

性能门禁（cambricon）：`ops.embedding.cambricon.forward` 0.060ms / 142.8 GB/s，
`ops.embedding.native.forward` 0.045ms / 187.9 GB/s；`perf/baselines/cambricon.json`
共 33 条 case。

## 结论与遗留

该需求适合“同名接口 + 厂商原生 row-gather 委托”的交付形态。与 SDPA 的
P800 结论一致：backend 存在不等于现有 Triton kernel 竞争力足够。本算子中
native embedding/index-select 已经是最优路径。

第二平台（cambricon）进一步印证：**原生算子能力决定实现复杂度**——MLU 的
`embedding_backward` 完整，MLU 后端因此是纯委托（零补偿代码）；XPU 缺
`scale_grad_by_freq`，P800 后端才需要 inverse-frequency 补偿。两平台共用同一
套三层测试与 174 条黄金数据，逐位判据一致。

## 交付物清单

| 内容 | 位置 |
|---|---|
| 需求 | [requirement.md](requirement.md) |
| 语义参考 | [reference.py](reference.py) |
| P800 backend | [kernel/p800_kunlunxin.py](kernel/p800_kunlunxin.py) |
| Cambricon backend | [kernel/cambricon.py](kernel/cambricon.py) |
| 多平台门面 | [kernel/platform.py](kernel/platform.py) |
| A1 注册 | [register.py](register.py) |
| 一键三层 | [example.py](example.py) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 开发报告 | [reports/development.md](reports/development.md) |
| cambricon 平台报告 | [reports/cambricon.md](reports/cambricon.md) |
| cambricon 性能原始数据 | [reports/perf_fp16_cambricon.json](reports/perf_fp16_cambricon.json) |
| cambricon dispatch 原始数据 | [reports/dispatch_cambricon.json](reports/dispatch_cambricon.json) |
| perf 门禁基线 | `perf/baselines/cambricon.json`（33 case，含本算子 2 条） |

## 遗留

1. `sparse=True` backward / sparse weight gradient 不在本期范围（MLU 侧显式
   `NotImplementedError`，避免 torch_mlu 返回 COO 稀疏梯度）；
2. `scale_grad_by_freq=True` 是能力补齐路径：P800 比 dense backward 慢约 47%，
   MLU 慢约 11%；
3. Triton gather 仅保留为平台证据，不作为生产实现（MLU 上 131k indices 还会
   撞 grid 65535 上限）；
4. **P800 的 A1 dispatch 数字作废待复测**：历史 `+0.0241ms` 出自
   `bench_dispatch.py` 未持有 `torch.library.Library` 的缺陷（析构即反注册，
   实测的是未注册状态），本机无 `torch_xmlir` 无法重测；native / direct 两行
   仍然有效。详见 [reports/cambricon.md](reports/cambricon.md) §7。
