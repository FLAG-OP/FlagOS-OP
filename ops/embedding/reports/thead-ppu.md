# thead-ppu embedding 移植与验证报告

日期：2026-10-23

## 1. 环境

| 项 | 值 |
|---|---|
| 容器 | `flagos-ppu-wt-dev`（flagtree-ppu-3.6-dev:202608，py3.12.3） |
| Torch | 2.10.0（CUDA 兼容层，16 卡 `cuda:0-15`） |
| Triton | 3.6.0（PPUDriver，target cuda arch=80） |
| vLLM | 0.19.0+cu130 |
| FlagGems / vllm-plugin-fl | **无**（dev 栈结构性缺失，见 §6） |
| 设备 | **PPU-ZW810E × 16**；用卡 **6,7**（避 4/5/11）；测试卡 `cuda:6` |
| profile | `thead-ppu`（`auto_detect.enabled=false`，显式 `EMBEDDING_PROFILE=thead-ppu` / argv；`EMBEDDING_TEST_DEVICE` 可覆盖测试卡） |

环境核对：

```bash
ppu-smi                 # PPU-ZW810E；确认 6/7 空闲
```

## 2. 首测结论（§2b，probes/thead_first_probe.py）

### 2.1 overload 与精度

| 探针 | 结果 |
|---|---|
| `aten::embedding` / `index_select` / `embedding_backward` schema | 均存在 |
| native `embedding` vs CPU 官方 | fp32/fp16/bf16 × padding ∈ {-1,3} **全部 err=0.0**（逐位） |
| `index_select` 委托面 | err=0.0 |
| `embedding_backward` `scale_grad_by_freq=False` | fp16 err=1.22e-4（vs CPU fp32 累加, fp16 累加序噪声） |
| `embedding_backward` `scale_grad_by_freq=True` | fp16 err=6.10e-5——**原生完整**，与 cambricon 同型，无需 p800/XPU 的 inverse-frequency 补偿 |

### 2.2 dispatch key（每 key 独立子进程，call count>0 才算命中）

| key | 命中探针 | 结论 |
|---|---|---|
| **AutogradCUDA** | grad_no_rg / grad_rg / no_grad / aten_direct 6/7（唯一零命中 inference_mode = known-issues #19 Autograd 键通病） | ✅ 采用（与 p800/T1-sdpa 先例一致） |
| CUDA | 7/7 全命中 | 备选；不取（须配对 Autograd 键才携带自带 backward，#19.3） |

### 2.3 实现选择

生产路径与 p800/cambricon **同构**（三方唯一差异收敛在 backward 补偿）：

```
业务 F.embedding → A1 AutogradCUDA
  → thead_ppu.embedding
       forward: aten::index_select row-gather（1D 快路径免 reshape;
                多 rank flatten→restore; 绕开 A1 递归）
  → thead_ppu.embedding_backward
       dense: aten::embedding_backward 直接委托（原生完整,
              scale_grad_by_freq/padding_idx 直通, 零补偿——cambricon 同型）
       sparse: 显式 NotImplementedError（守卫口径与两平台一致）
```

Triton gather 不接生产：P800/MLU 两平台实测慢 11-171x（README Triton
实验节），embedding 是带宽型算子；T1 实测 PPU triton `tl.dot` 精度掉到
fp16 量级（sdpa 报告 §2.1），gather 场景更无理由绕开 native。

## 3. A1 注册与路由

| 项 | 结论 |
|---|---|
| dispatch key | **`AutogradCUDA`**（§2.2）；`_profile.py` thead-ppu 条目同 |
| cuda 双候选路由 | `kernel/platform.py::_cuda_impl` + `register.py::_cuda_backend`：torch_xmlir → p800，PPU 设备名 → thead，进程级缓存（T1 教训: 无缓存 find_spec 路由 80-100µs/调用） |
| 拦截 | op 层 count=5；注册路径与直调路径**逐位一致**；framework 拦截 8 次 |
| 梯度 | dense/scale_freq grad vs CPU 参考 err=0.0；应用层 embedding/首层 linear 梯度 diff=0 |
| 守卫 | sparse backward NotImplementedError / invalid padding IndexError——kernel 层 guard 2/2 通过（cambricon 先例同口径） |

## 4. 验证结果

```bash
PYTHONPATH=/hosthome/FlagOS-OP python3 example.py thead-ppu          # 三层一键
PYTHONPATH=/hosthome/FlagOS-OP python3 test/kernel_level_tail_thead_ppu.py thead-ppu
PYTHONPATH=/hosthome/FlagOS-OP python3 script/gen_golden.py
PYTHONPATH=/hosthome/FlagOS-OP python3 script/check_accuracy.py --impl thead-ppu --device cuda:6
PYTHONPATH=/hosthome/FlagOS-OP python3 scripts/perf_run.py --device thead-ppu --pattern ops.embedding --update-baseline
PYTHONPATH=/hosthome/FlagOS-OP python3 scripts/perf_compare.py --device thead-ppu
```

| 项 | 结果 |
|---|---|
| kernel 层 | ✅ 20 前向 + 6 反向，max_err **0.0**（逐位）；守卫 2/2；16k×D128 ~0.041ms |
| tail 覆盖（§2d） | ✅ 37/37：gather 长度 3072/5000/5120 × 3 dtype × 3 seed 全 err=0；zeros/large/boundary 全 err=0（gather 天然逐位, boundary 相对容差 5e-3 口径下 rel=0）；哨兵确定性+输入敏感通过 |
| 黄金 | ✅ **174/174**，worst=0.000e+00（逐位判据, CPU goldendata 直接复用） |
| op 层 | ✅ AutogradCUDA 拦截 5 次；注册=直调(逐位)；dense/scale grad err=0.0；sparse fwd OK / bwd rejected |
| 应用层 | ✅ nn.Embedding+TokenMLP bf16 拦截 8 次；logits diff=0.0；贪心一致率 1.00；两级梯度 diff=0 |
| perf gate | ✅ `ops.embedding.thead-ppu.forward` 0.033ms / 255.0 GB/s 入库；`perf_compare` FAIL 0 · WARN 0 |
| 仓库一致性 | ◐ `run.py --consistency` FAIL：`ModuleNotFoundError: flag_gems`（A2/gelu_and_mul 锚定门禁，dev 栈结构性缺证，同 T1 sdpa §6.11） |
| 单测 | ✅ `pytest tests/unit -q` 46 passed |
| 索引校验 | ✅ ops/env/base-image 三索引 --check 全绿 |
| 包化 | ◐ `ops/test_packaging.py` 需 torch_npu（dev 栈无，环境既有缺口非本改动引入；任务书 §4 三算子收口时在有 npu 栈环境补跑） |

## 5. 性能

数据：[perf_fp16_thead-ppu.json](perf_fp16_thead-ppu.json)、
[dispatch_thead-ppu.json](dispatch_thead-ppu.json)、
`perf/baselines/thead-ppu.json`（warmup=5，iters=20，每次读取一个输出
元素强制完成）。

### Forward（fp16）

| shape | ours | native | ours/native |
|---|---:|---:|---:|
| 1k×D128 | 0.0286ms | 0.0239ms | 0.833x |
| 16k×D128 | 0.0359ms | 0.0307ms | 0.855x |
| 131k×D128 | 0.1033ms | 0.0982ms | 0.951x |
| 16k×D512 | 0.0359ms | 0.0307ms | 0.855x |
| vocab128k 16k×D128 | 0.0360ms | 0.0275ms | 0.764x |

### Backward（16k×D128，fp16）

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.1669ms | 0.1597ms | 0.957x |
| scale_grad_by_freq | 0.1842ms | 0.1771ms | 0.962x |

### 仓库 perf gate

```text
ops.embedding.thead-ppu.forward   0.033ms · 254.97 GB/s
ops.embedding.native.forward      0.030ms · 286.90 GB/s
perf_compare: FAIL 0 · WARN 0（含 T1 sdpa 4 条共 6 条全 OK）
```

### A1 dispatch overhead（独立子进程）

| mode | median | p20 | p80 |
|---|---:|---:|---:|
| native `aten::embedding` | 0.0281ms | 0.0278ms | 0.0283ms |
| direct backend | 0.0315ms | 0.0312ms | 0.0318ms |
| A1 `F.embedding` | 0.0441ms | 0.0432ms | 0.0448ms |

A1 − native ≈ **+0.016ms**、A1 − direct ≈ **+0.013ms**——比 MLU 的
+0.059ms 更低（PPU 栈 autograd 往返更便宜）；与 native 的 0.76-0.95x
委托比落在 p800（0.89-1.00x）/cambricon（0.65-1.12x）同一带宽。

## 6. 平台差异与缺证

1. **原生 backward 完整性属 PPU 原生栈**：`scale_grad_by_freq=True`
   直接委托即可（cambricon 同型）；若未来栈升级出现 p800 式
   `Check scale_grad_by_freq == false failed`，回退 p800 的
   inverse-frequency 补偿路径（`kernel/p800_kunlunxin.py` 已有实现）。
2. **Triton gather 缺证**：dev 栈无 flag_gems，`bench_perf.py` 的
   triton 列 `ModuleNotFoundError`（与 T1 sdpa 的 FlagGems 三方对比
   缺证同口径——dev 栈结构性，非 thead/embedding 缺陷）。P800/MLU 已有
   两平台 Triton 慢 11-171x 的先例，不影响生产路径结论。
3. **FlagOS 栈应用层缺证**：framework_level 在 thead 下跳过
   `flag_gems.only_enable(['gelu'])` 门禁（纯 A1 拦截口径），拦截/数值/
   梯度/贪心全部与基线一致。
4. **`run.py --consistency`** 锚定 A2/gelu_and_mul 需要 flag_gems，
   dev 容器结构性 FAIL——同 T1 sdpa 报告 §6.11，不计入本算子缺陷。
5. **测试尾块维解释**：§2d 的 3072/5000/5120 对 embedding 取 gather
   长度维（#15a 尾块 masked-load 归约陷阱不适用——gather 无归约）；
   全 err=0 逐位通过。

## 7. 复现命令

```bash
export PYTHONPATH=/hosthome/FlagOS-OP
cd ops/embedding

# 首测探针（已入库 probes/out/）
python3 probes/thead_first_probe.py
python3 probes/thead_first_probe.py --key AutogradCUDA

# 三层 + 尾块
python3 example.py thead-ppu
python3 test/kernel_level_tail_thead_ppu.py thead-ppu

# 黄金
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl thead-ppu --device cuda:6

# 性能
python3 script/bench_perf.py --device cuda:6 \
  --json-out reports/perf_fp16_thead-ppu.json
python3 script/bench_dispatch.py --device cuda:6 \
  --json-out reports/dispatch_thead-ppu.json
python3 scripts/perf_run.py --device thead-ppu --pattern ops.embedding
python3 scripts/perf_compare.py --device thead-ppu
```
