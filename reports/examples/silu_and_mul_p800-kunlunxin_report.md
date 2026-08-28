# 算子开发报告: silu_and_mul

| 项 | 值 |
|---|---|
| 算子名称 | `silu_and_mul` |
| 实现路线 | B |
| 目标设备 | `p800-kunlunxin` |
| 开发者 | `<手填>` |
| 日期 | `<手填>` |
| 报告状态 | 草稿 |

---

## 1. 环境配置

> 自动生成: `scripts/env_snapshot.py --device p800-kunlunxin`；标注 `[手填]` 的项需人工补充。

### 1.1 硬件

设备 profile: **p800-kunlunxin**（vendor=kunlunxin，torch_device=cuda:1，visible=1,2）

```
    Mon Aug 24 21:19:00 2026       
    +-----------------------------------------------------------------------------+
    | XPU-SMI               Driver Version: 515.58       XPU-RT Version: N/A      |
    |-------------------------------+----------------------+----------------------+
    | XPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | XPU-Util  Compute M. |
    |                               |             L3-Usage |            SR-IOV M. |
    |===============================+======================+======================|
    |   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |                    0 |
    | N/A   39C  N/A     91W / 400W |  31058MiB / 98304MiB |      0%      Default |
    |                               |      0MiB /    96MiB |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   1  P800 OAM           N/A   | 00000000:16:00.0 N/A |                    0 |
    | N/A   38C  N/A     94W / 400W |      0MiB / 98304MiB |      0%      Default |
    |                               |      0MiB /    96MiB |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   2  P800 OAM           N/A   | 00000000:1C:00.0 N/A |                    0 |
    | N/A   44C  N/A     93W / 400W |      0MiB / 98304MiB |      0%      Default |
    |                               |      0MiB /    96MiB |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   3  P800 OAM           N/A   | 00000000:2E:00.0 N/A |                    0 |
    | N/A   44C  N/A     94W / 400W |      0MiB / 98304MiB |      0%      Default |
    |                               |      0MiB /    96MiB |             Disabled |
    +-------------------------------+----------------------+----------------------+
```

拓扑 / 卡号映射 `[手填]`

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | Linux 6.8.0-87-generic |
| Python | 3.10.18 |
| PyTorch | 2.9.0+cu129 |
| vLLM | 0.13.0 |
| FlagGems | 4.2.1.rc.0 |
| vllm-plugin-FL | unknown |
| Triton | 3.0.0 |
| transformers | 4.57.1 |
| 厂商库 | [手填] |

### 1.3 设备 profile

```yaml
# 设备 Profile: Kunlunxin P800 (本机实测 case)
name: p800-kunlunxin
vendor: kunlunxin

auto_detect:
  module: torch_xmlir        # 可 import 即认定为该设备

device:
  torch_device: "cuda:1"     # 算子层测试设备 (XPU 经 XMLIR 伪装为 CUDA)
  visible_devices: "1,2"     # 框架级测试暴露的卡
  dispatch_key: CUDA         # A1 aten 注册的 dispatch key

framework:
  # 框架级 vLLM 引擎参数（本机已验证的稳定组合）
  tensor_parallel_size: 2
  distributed_executor_backend: mp
  max_model_len: 4096
  max_num_seqs: 50
  max_num_batched_tokens: 8192
  gpu_memory_utilization: 0.85
  enforce_eager: true
  enable_prefix_caching: false
  load_format: safetensors
  seed: 20260823
  model_path: /workspace/random-llm-models/llama-3.1-8b-like
  quirks:
    keep_default_prefer: true        # 禁止覆盖 VLLM_FL_PREFER（容器默认非法值是保护路径）
    require_two_visible_devices: true # 单卡路径曾出现厂商 reshape_and_cache 通道异常

vendor_delegate:
  # audit vendor 的 silu_and_mul 委托目标（厂商 kernel 原样透传）
  silu_and_mul: xtorch_ops.swiglu

reference_fallback: reference.torch

# 厂商硬件语言 kernel 直测清单（kernel 层 / --level kernel）
vendor_kernels:
  - name: xtorch_ops.gelu_tanh_and_mul
    op: gelu_and_mul
    out_mode: return
    adapter: gelu_tanh_and_mul
    n_inputs: 2
    expected_ok: false
    known_issue: XPU2(=cuda:1) 上非确定且输出 inf; XPU1(=cuda:0) 上 err≈0.125 正常（按设备分化 bug，known-issues #10）
  - name: xtorch_ops.silu
    op: silu
    out_mode: out_param
    adapter: out_param
    n_inputs: 1
    expected_ok: false
    known_issue: out 参数不写输出（与 swiglu 同类，known-issues #2）
  # 已知坏例: 哨兵检查应能抓出（验证检测能力）
  - name: xtorch_ops.swiglu
    op: gelu_and_mul
    out_mode: out_param
    adapter: out_param
    n_inputs: 2
    expected_ok: false
    known_issue: 独立 eager 调用不写输出（known-issues #2）
  # 自研 C++ kernel JIT 编译产物（csrc 闭环）
  - name: my_vendor_ops.gelu_and_mul
    op: gelu_and_mul
    out_mode: return
    adapter: two_tensor
    n_inputs: 2
    build: csrc
```

### 1.4 关键环境变量（采集时）

| 变量 | 值 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `1,2` |
| `VLLM_FL_PREFER` | `flagos|vendor` |
| `VLLM_FL_PLATFORM` | `kunlunxin` |
| `USE_FLAGGEMS` | `1` |
| `VLLM_FL_PER_OP` | `(未设置)` |
| `VLLM_FL_PLUGIN_MODULES` | `(未设置)` |

### 1.5 环境特殊性说明 `[手填]`

<容器限制 / 共享设备 / quirks 开关原因>

---

## 2. 算子定义 `[手填]`

### 2.1 语义

<一段话 + 公式/伪代码>

### 2.2 PyTorch 参考实现

```python
# <手填>
```

### 2.3 数值规格

<dtype / 内部精度 / 容差>

---

## 3. 实现说明 `[手填]`

- 代码位置: `routes/... / examples/...`
- kernel 要点: <手填>
- 注册与分发: <impl_id / 优先级 / PER_OP 钉选>

---

## 4. 验证结果

| 层级 | 状态 | 耗时 | 关键结论 `[手填]` |
|---|---|---|---|
| kernel 直测 | ✅ PASS | 2.3s | <手填> |
| op 注册/分发 | ✅ PASS | 7.1s | <手填> |
| framework 真实推理 | ✅ PASS | 98.9s | <手填> |

### 4.1 kernel 层明细 `[手填]`

| shape | dtype | vs 参考 max_err | 哨兵 | 性能 |
|---|---|---|---|---|
| | | | | |

### 4.2 framework 层明细 `[手填]`

- 调用计数: <从测试输出抄录>
- 输出比对: <模式 + 结果>

---

## 5. 性能 `[手填]`

| 实现 | 耗时/call | 有效带宽 | 相对参考加速 |
|---|---|---|---|
| 本实现 | | | |
| PyTorch 参考 | | | 1x |

> 基准方法: 短采样（≤100 次），避免分配器池增长失真。

---

## 6. 已知问题与风险 `[手填]`

| # | 问题 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| 1 | | | | |

---

## 7. 结论与后续 `[手填]`

### 结论

<是否达到验收标准>

### 后续

- [ ]

---

## 附录 A: 复现命令

```bash
python3 run.py --route b --level kernel --device p800-kunlunxin
python3 run.py --route b --level op --device p800-kunlunxin
python3 run.py --route b --level framework --device p800-kunlunxin
python3 run.py --consistency --device p800-kunlunxin
python3 scripts/report.py --device p800-kunlunxin
```
