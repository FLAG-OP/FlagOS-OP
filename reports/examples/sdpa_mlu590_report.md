# 算子开发报告: sdpa @ mlu590

| 项 | 值 |
|---|---|
| 算子名称 | `aten::scaled_dot_product_attention` |
| 实现路线 | **A1** torch 算子替换 |
| 开发级别 | 厂商 kernel 委托（TMO FA + fused overrideable） |
| 目标设备 | `mlu590` |
| 开发者 | FlagOS Dev |
| 日期 | 2026-09-23 |
| 报告状态 | 定稿 |

---

## 1. 环境配置

> 自动生成: `scripts/env_snapshot.py --device mlu590`；
> `scripts/check_env.py --device mlu590` **OK**。

### 1.1 硬件

| 项 | 值 |
|---|---|
| 设备型号 | **Cambricon MLU590-M9** ×8（CNMON v6.5.48，Driver v6.2.29，FW v1.5.0，96 GiB/卡） |
| 测试卡 | `MLU_VISIBLE_DEVICES=7` → 逻辑设备 `mlu:0`（Card 7，`0000:D3:00.0`） |
| 驱动/运行时 | CNRT 7.7.2 · CNNL 2.2.14 · CNNLExtra 2.4.0 · CNCL 1.30.8 |

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | Linux 5.15.0-139-generic |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cpu |
| torch_mlu | 1.33.1+torch2.11.0 |
| torch_mlu_ops (TMO) | 与 torch_mlu 同栈；`flash_attention` → CNNL SDPA v7 |
| FlagGems | 5.3.5（`_cambricon`） |
| Triton | 3.4.0+mlu2.1.1 |
| vLLM | 未安装（profile `no_vllm_runtime=true`） |

锁定文件: [configs/env/mlu590.lock.yaml](../../configs/env/mlu590.lock.yaml)。

### 1.3 设备 profile 与关键环境变量

```yaml
# 见 configs/devices/mlu590.yaml
name: mlu590
vendor: cambricon
auto_detect: { module: torch_mlu }
device:
  torch_device: "mlu:0"
  dispatch_key: AutogradPrivateUse1
framework.quirks.no_vllm_runtime: true
```

```bash
MLU_VISIBLE_DEVICES=7          # 本机测试卡
SDPA_MLU_TMO=0                 # 可选: 关闭 TMO，只走 fused overrideable
```

### 1.4 环境特殊性说明

- torch 为 CPU 轮子 + torch_mlu 插件；非 CUDA 构建。
- 无 vLLM / transformers / vllm-plugin-FL —— 应用层用仓库
  mini-decoder harness，不走 `run.py --level framework` 引擎入口。
- 共享容器 MLU 卡；测试前用 `MLU_VISIBLE_DEVICES` 钉卡。

---

## 2. 算子定义

### 2.1 语义

缩放点积注意力（flash-attention 族）:

```
out = softmax(Q @ K^T * scale + mask_bias [+ causal]) @ V
```

### 2.2 PyTorch 参考实现

见 [ops/sdpa/reference.py](../../ops/sdpa/reference.py)
（fp32 内部、GQA、causal、bool/float mask；**不**调用 `F.sdpa`，
与被测实现解耦）。黄金生成时与 CPU `F.sdpa` 交叉互验。

### 2.3 接口签名

| 形态 | 签名 |
|---|---|
| aten（A1） | `aten::scaled_dot_product_attention(...)` |
| Python 业务入口 | `F.scaled_dot_product_attention` → A1 拦截 |
| backend 直调 | `kernel.backends.mlu590.sdpa_triton(q,k,v,...)` |

### 2.4 数值规格

| 项 | 值 |
|---|---|
| 支持 dtype | fp16 / bf16 / fp32 |
| 内部计算 | 厂商 fused/TMO（fp32 累加语义） |
| 容差 | fp32: 1e-5 abs；fp16/bf16: 2e-2 abs |

---

## 3. 实现说明

### 3.1 路线选择理由

- MLU 有成熟 CNNL FA kernel，自写 Triton 无优势（对照 P800 结论）。
- **关键更正**: 原生 `F.sdpa` → `_scaled_dot_product_fused_attention_
  overrideable`（CNNL FA v2），**不是** math。早期 math-only 主路径
  仅 0.11–0.55x 原生。
- efficient/flash/cudnn 三个 aten private op 不可用（CPU fallback）。

### 3.2 实现要点（`kernel/backends/mlu590.py`）

```
TMO fast path（fp16/bf16, SDPA_MLU_TMO=0 可关）
  → fused overrideable（P0 / fp32 / TMO 失败兜底）
  → math private → FlagGems → fp32 reference
```

- bool mask → additive `-inf` bias；causal+mask 先折叠
- 全遮蔽行输出端 `masked_fill` 恢复 NaN
- GQA: overrideable 无 `enable_gqa` → `repeat_interleave`；TMO 在
  BSHD head 维扩展；TMO bias 形状 BHSD `(B,H,Sq,Skv)`
- 可微直调走 `_SDPA_A1_Function` 数学 backward（TMO/OV 无完整 autograd）

### 3.3 注册与分发

- facade: `kernel/triton_level.py` 按 `query.device.type` 分发
  `"mlu" → mlu590`
- A1: `AutogradPrivateUse1`（同 torch_npu 栈）；守卫要求 torch_mlu
  可导入；`probes/guard_check.py mlu590` 全绿
- 业务代码零改动：仍只调 `F.sdpa`

---

## 4. 验证结果

| 物理栈层 | 验证层级 | 状态 | 关键结论 |
|---|---|---|---|
| 算子库层 | kernel 直测 | ✅ | 37/37，max_err 3.91e-3（bf16），哨兵过，1k D128 0.25ms |
| 框架层 | op 注册/分发 | ✅ | 拦截 2；注册=直调逐位；vs 原生 4.88e-4；梯度 ≤6.1e-5 |
| 应用层 | framework 消费 | ✅ | mini-decoder 拦截 28；logits 1.95e-3；续写一致 1.00 |
| 黄金 | 黄金回归 | ✅ | 397/397，worst 7.81e-3 |
| 守卫 | 平台守卫 | ✅ | 元数据 / mlu / CPU 拒绝 / 注册 |
| 基础设施 | pytest | ✅ | 20 passed |

明细见 [测试报告](sdpa_mlu590_test_report.md) 与
[reports/mlu590.md](../../ops/sdpa/reports/mlu590.md)。

### 4.1 kernel 层明细（摘要）

| shape 族 | dtype | max err | 哨兵 | 性能 |
|---|---|---|---|---|
| basic / GQA / mask / tail | fp32 | 1.79e-7 | ✓ | — |
| 同上 | fp16 | 4.88e-4 | ✓ | — |
| 同上 | bf16 | 3.91e-3 | ✓ | 1k D128 0.252ms |

### 4.2 framework 层明细

- 调用计数: intercept **28**（4 层 × prompt）
- 输出比对: logits max_diff 1.95e-3（<0.05）· top-1 1.00 · 贪心续写 1.00

---

## 5. 性能

| 实现 | prefill 1k D64 | prefill 4k D128 | 相对原生 |
|---|---:|---:|---|
| **本实现**（TMO→fused） | 0.227ms | 0.727ms | 1.00x 基准见下 |
| PyTorch 原生 F.sdpa | 0.301ms | 0.902ms | — |
| FlagGems `_cambricon` | 2.915ms | 26.628ms | 慢 10–40x |

相对原生 speedup: 1k D64 **1.33x** · 1k D128 **1.28x** · 2k **1.27x** ·
4k **1.24x** · GQA **1.21x** · decode **1.00x**。

> 基准: 短采样 warmup=20 + iters=100 + MLU sync；JSON 落盘
> `ops/sdpa/reports/perf_fp16_mlu590.json`。

---

## 6. 已知问题与风险

| # | 问题 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| 1 | aten efficient/flash/cudnn 不可用 | 无 efficient 入口 | fused OV + TMO | [known-issues #18](../../docs/known-issues.md) |
| 2 | math bool mask err≈0.19 | 精度 | 主路径避开 math；bool→additive | 已缓解 |
| 3 | FlagGems 慢 10-40x | 不能作主路径 | 仅兜底 | 已隔离 |
| 4 | 无 vLLM runtime | 真实引擎未测 | mini-decoder + quirks | 有条件 |
| 5 | TMO 无 autograd | 训练 | A1 数学 backward | 已缓解 |

---

## 7. 结论与后续

### 结论

三层 + 黄金 + 守卫 + 单测 + 环境锁全绿；性能 **1.00–1.33x** 原生
（math-only 0.11–0.55x 已废弃）。**达到验收标准，可合入**。

### 后续

- [ ] 有 vLLM 的环境补跑 `run.py --route a1 --level framework`
- [x] fused/TMO 路径性能修订（本报告）
- [ ] （可选 P2）`mlu_use_tmo_fa` 开关对比；`torch.compile` 已证本栈无效

---

## 附录 A: 复现命令

```bash
cd ops/sdpa
python3 example.py mlu590
python3 test/kernel_level.py mlu590
python3 script/check_accuracy.py --impl triton --device mlu:0
python3 test/op_level.py mlu590
python3 test/framework_level.py mlu590
python3 probes/guard_check.py mlu590
python3 script/bench_perf.py --device mlu:0 --json-out reports/perf_fp16_mlu590.json
python3 ../../scripts/check_env.py --device mlu590
python3 -m pytest ../../tests/unit -q
SDPA_MLU_TMO=0 python3 test/kernel_level.py mlu590
```

## 附录 B: 相关产物

| 产物 | 路径 |
|---|---|
| backend | `ops/sdpa/kernel/backends/mlu590.py` |
| 设备 profile | `configs/devices/mlu590.yaml` |
| 环境锁 | `configs/env/mlu590.lock.yaml` |
| 平台复现报告 | `ops/sdpa/reports/mlu590.md` |
| 性能 JSON | `ops/sdpa/reports/perf_fp16_mlu590.json` |
| FA2 交叉 | `ops/sdpa/reports/cross_platform_mlu_0.json` |
| 测试报告 | `reports/examples/sdpa_mlu590_test_report.md` |
