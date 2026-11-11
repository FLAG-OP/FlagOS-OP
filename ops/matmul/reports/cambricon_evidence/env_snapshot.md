## 1. 环境配置

> 自动生成: `scripts/env_snapshot.py --device mlu590`；标注 `[手填]` 的项需人工补充。

### 1.1 硬件

设备 profile: **mlu590**（vendor=cambricon，torch_device=mlu:0，visible=0）

```
    (厂商 cambricon 无自动探测命令，请手填)
```

拓扑 / 卡号映射 `[手填]`

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | Linux 5.15.0-139-generic |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cpu |
| vLLM | (未安装) |
| FlagGems | unknown |
| vllm-plugin-FL | (未安装) |
| Triton | 3.4.0 |
| transformers | 4.57.1 |
| 厂商库 | [手填] |

### 1.3 设备 profile

```yaml
# 设备 Profile: Cambricon MLU590 (本机实测 case)
name: mlu590
vendor: cambricon                 # 与 flag_gems DeviceDetector().vendor_name 一致

auto_detect:
  module: torch_mlu               # 可 import 即认定为该设备

device:
  torch_device: "mlu:0"           # 算子层测试设备
  visible_devices: "0"
  dispatch_key: AutogradPrivateUse1  # torch_mlu 实测拦截点（同 torch_npu）

framework:
  tensor_parallel_size: 1
  distributed_executor_backend: mp
  max_model_len: 4096
  max_num_seqs: 50
  max_num_batched_tokens: 8192
  gpu_memory_utilization: 0.85
  enforce_eager: true
  enable_prefix_caching: false
  load_format: safetensors
  seed: 20260823
  model_path: /path/to/model
  quirks:
    keep_default_prefer: false
    require_two_visible_devices: false
    no_vllm_runtime: true         # 本环境 vllm 为空壳时跳过框架/应用层入口

vendor_delegate: null

reference_fallback: reference.torch

# 原生参照走 torch_mlu 的 F.scaled_dot_product_attention
# → aten::_scaled_dot_product_fused_attention_overrideable（CNNL FA v2）
vendor_kernels: []
```

### 1.4 关键环境变量（采集时）

| 变量 | 值 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `(未设置)` |
| `VLLM_FL_PREFER` | `(未设置)` |
| `VLLM_FL_PLATFORM` | `(未设置)` |
| `USE_FLAGGEMS` | `(未设置)` |
| `VLLM_FL_PER_OP` | `(未设置)` |
| `VLLM_FL_PLUGIN_MODULES` | `(未设置)` |

### 1.5 环境特殊性说明 `[手填]`

<容器限制 / 共享设备 / quirks 开关原因>
