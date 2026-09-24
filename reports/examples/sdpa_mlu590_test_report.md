# 算子测试报告: sdpa @ mlu590（已填样例）

| 项 | 值 |
|---|---|
| 算子名称 | `aten::scaled_dot_product_attention` |
| 实现路线 | **A1** torch 算子替换（aten 拦截） |
| 开发级别 | 厂商 kernel 委托（TMO FA + fused overrideable；非自研 Triton） |
| 目标设备 | `mlu590` |
| 测试日期 | 2026-09-23 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 是否覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ✅ | `ops/sdpa/test/kernel_level.py mlu590` | PASS（**37/37** + 哨兵 + 1k D128 0.25ms） |
| 框架层（注册/拦截/梯度） | ✅ | `ops/sdpa/test/op_level.py mlu590` | PASS（A1 拦截 · 注册=直调逐位 · vs 原生 4.88e-4 · 梯度 ≤6.1e-5） |
| 应用层（消费方双跑） | ✅ | `ops/sdpa/test/framework_level.py mlu590` | PASS（mini-decoder 拦截 28 · logits 1.95e-3 · 续写一致率 1.00） |
| 黄金回归 | ✅ | `script/check_accuracy.py --impl triton --device mlu:0` | PASS（**397/397**，worst 7.81e-3 @ bf16 bool） |
| 性能三方对照 | ✅ | `script/bench_perf.py --device mlu:0` | OK（fp16 6 形状，**1.00–1.33x** 原生） |
| 平台守卫（三层） | ✅ | `probes/guard_check.py mlu590` | PASS（元数据 / mlu 路径 / CPU 拒绝 / 注册） |
| 环境锁定 | ✅ | `scripts/check_env.py --device mlu590` | OK（lock 对齐） |
| 基础设施单测 | ✅ | `pytest tests/unit -q` | PASS（**20 passed**） |
| 一键复跑 | ✅ | `python3 example.py mlu590` | 三层全绿 |

跨层一致性 / 性能回归门禁（`run.py --consistency` / `perf_compare`）:
本环境无 vLLM（profile `no_vllm_runtime=true`），应用层以
`framework_level.py` mini-decoder 双跑替代；性能门禁以
`bench_perf.py` + 仓库 perf JSON 落盘替代。

## 2. 环境

`check_env --device mlu590` **OK**；快照 `env_snapshot --device mlu590`。

| 组件 | 版本 |
|---|---|
| OS / 内核 | Linux 5.15.0-139-generic |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cpu（+ torch_mlu 1.33.1 插件加速） |
| torch_mlu / TMO | 1.33.1+torch2.11.0 / `flash_attention`→CNNL SDPA v7 |
| FlagGems | 5.3.5（`_cambricon`） |
| Triton | 3.4.0+mlu2.1.1 |
| CNNL / CNRT | 2.2.14 / 7.7.2 |
| 设备 | MLU590-M9 ×8（CNMON v6.5.48 / Driver v6.2.29），`MLU_VISIBLE_DEVICES=7` → `mlu:0` |
| vLLM | 未安装（quirks 跳过框架层 vLLM 入口） |

设备 profile 与锁定: [configs/devices/mlu590.yaml](../../configs/devices/mlu590.yaml) ·
[configs/env/mlu590.lock.yaml](../../configs/env/mlu590.lock.yaml)。

## 3. 精度结果

口径: 对 `ops/sdpa/reference.py`（fp32 内部）**绝对误差**；
容差 fp32=1e-5 · fp16/bf16=2e-2 · extreme 相对 5e-3。

### kernel 层直测（37 组）

| dtype | max err | 容差 | 判定 |
|---|---|---|---|
| fp32 | 1.79e-7 | 1e-5 | ✓ |
| fp16 | 4.88e-4 | 2e-2 | ✓ |
| bf16 | 3.91e-3 | 2e-2 | ✓ |

shape 覆盖: basic/batch/尾块 100·333/ViT D=80/GQA/bool/float mask/
Sq=1 decode。哨兵: 确定性 ✓ · 输入敏感 ✓。

### 黄金回归（397 组）

| 实现 | PASS | worst |
|---|---|---|
| 自研路径（TMO→fused→…） | **397/397** | 7.81e-3（`basic_bfloat16_..._mbool`，容差内） |

### 框架层（A1 注册后）

- 拦截 count=**2**；注册路径与直调路径**逐位一致**
- vs 未注册原生 `F.sdpa`: max_diff **4.88e-4**
- 梯度 vs 原生独立进程: dq 1.53e-5 · dk 6.10e-5 · dv 0

## 4. 性能结果

fp16 · warmup=20 · iters=100（`reports/perf_fp16_mlu590.json`）；
speedup = native/ours，**>1 自研更快**。路径: 半精度 TMO FA →
fused overrideable（`SDPA_MLU_TMO=0` 可关 TMO）。

| shape | 自研 | PyTorch 原生 F.sdpa | FlagGems | 相对自研（原生） |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.227ms | 0.301ms | 2.915ms | **1.33x** |
| prefill 1k D128 | 0.283ms | 0.363ms | 3.666ms | **1.28x** |
| prefill 2k D128 | 0.351ms | 0.445ms | 8.088ms | **1.27x** |
| prefill 4k D128 | 0.727ms | 0.902ms | 26.628ms | **1.24x** |
| GQA 1k D128 | 0.366ms | 0.442ms | 5.176ms | **1.21x** |
| decode D128 | 0.225ms | 0.224ms | 1.989ms | **1.00x** |

FA2 协议 TFLOPS（ours 多点高于 native）: D64 S=4096 **72.3 vs 48.5**；
D128 S=4096 **115.7 vs 95.6**（见 `cross_platform_mlu_0.json`）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | efficient/flash/cudnn aten private op CPU fallback 失败 | 只能走 fused overrideable / TMO | 已记录 [known-issues #18](../../docs/known-issues.md) |
| 2 | math private 误处理 bool mask（err≈0.19） | 主路径已避开 math；bool→additive bias | 已缓解 |
| 3 | FlagGems `_cambricon` 慢 10-40x 且无 autograd | 仅兜底/对照 | 已隔离 |
| 4 | 全遮蔽行 fused/TMO 不返回 NaN | 输出端 `masked_fill` 恢复 | 已缓解 |
| 5 | 本环境无 vLLM | 无法跑 `run.py --level framework` 真实推理 | mini-decoder 替代；quirks 已声明 |
| 6 | TMO 无完整 Python autograd | 可微直调走 A1 数学 backward | 已缓解 |

## 6. 结论

三层 + 黄金 + 守卫 + 单测 + 环境锁全绿；fp16 相对原生 **1.00–1.33x**
（早期 math-only 路径 0.11–0.55x 已废弃）。**可合入**。

遗留: P2 可选（`torch.compile` 本栈已证无效；`mlu_use_tmo_fa` 未探）；
vLLM 应用层待有 runtime 的环境补跑。
