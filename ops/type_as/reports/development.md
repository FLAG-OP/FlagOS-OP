# 算子开发报告: type_as

| 项 | 值 |
|---|---|
| 算子名称 | `type_as` |
| 实现路线 / 开发级别 | **A1** torch 算子替换 / **Triton 级**（+ torch 级对照） |
| 目标设备 | `cambricon`（Cambricon MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 1. 环境配置

### 1.1 硬件

| 项 | 值 |
|---|---|
| 设备型号 | Cambricon MLU590-M9，8 卡，单卡 98304 MiB |
| 固件 / 驱动 | firmware v1.5.0 / CNMON Driver v6.2.29 |
| 拓扑 | 本算子验证仅用 `mlu:0`，未做多卡 |

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | Linux 5.15.0-139-generic |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cpu（+ torch_mlu 设备扩展） |
| torch_mlu | 1.29.2+torch2.7.1 |
| Triton | 3.2.0+mlu1.7.2 |
| FlagGems | 5.3.5 |
| vLLM / vllm-plugin-FL / transformers | 未安装 |

### 1.3 设备 profile 与关键环境变量

见 [configs/devices/cambricon.yaml](../../../configs/devices/cambricon.yaml)
（`vendor: cambricon`，`torch_device: mlu:0`，`dispatch_key: PrivateUse1`；
`auto_detect.module: torch_mlu` 可自动探测）。关键环境变量均未设置，走默认。

### 1.4 环境特殊性说明

- 本机为 **CPU 版 torch + torch_mlu 设备扩展**，MLU 是 PyTorch
  `PrivateUse1` 后端（`torch._C._get_privateuse1_backend_name() == "mlu"`）。
- MLU 栈要求 **先 `import torch` 再导入 triton**，否则
  `torch.library` 的 `triton` 命名空间会重复注册报错。为此在
  `common/device.py` 顶部补了 `import torch`（早于 `xpu_compat`→triton）。
- 未安装 vLLM/transformers，应用层用等价的 torch API 子进程验证
  （见 §4 与 [reports/test-report.md](test-report.md)）。

## 2. 算子定义

### 2.1 语义

`aten::type_as(Tensor self, Tensor other) -> Tensor`：把 `self` cast 到
`other.dtype`，结果始终留在 `self.device`；`other` 的 shape/strides/device
被忽略，只取 dtype。`self.dtype == other.dtype` 时返回 `self` 本身
（别名存储）；否则输出 layout 沿用 `to()` 的 `preserve_format` 决策
（non-overlapping & dense 保留 stride，overlapping/stepped 走 dense）。

```
out = self.to(other.dtype)          # 无算术放大，cast = IEEE round-to-nearest
```

### 2.2 PyTorch 参考实现

见 [reference.py](../reference.py)：`type_as_reference(self, other) = self.to(other.dtype)`。

### 2.3 接口签名

| 形态 | 签名 |
|---|---|
| aten（A1 宿主） | `aten::type_as(Tensor self, Tensor other) -> Tensor` |
| Triton 级 | `type_as_triton(self, other) -> Tensor` |

### 2.4 数值规格

| 项 | 值 |
|---|---|
| 支持 dtype | bf16 / fp16 / fp32（其余走原生回退） |
| 内部计算精度 | 无（纯 cast，位级可复现） |
| 容差 | 输出 fp32: 1e-5；fp16/bf16: 1e-2（本实现实测 **0**） |
| shape 清单 | 含非 BLOCK 整倍数 14336 / 5000 / 5120 / 4097 |

## 3. 实现说明

### 3.1 路线选择理由

`type_as` 是标准 aten 算子，宿主是 torch aten dispatcher，按
[集成指南](../../../docs/integration.md) 属 **A1**。注册后任何框架代码
（vLLM/transformers/自研）调 `Tensor.type_as` 自动命中，业务零改动。

### 3.2 kernel 实现要点

代码: [kernel/triton_level.py](../kernel/triton_level.py)。

- 同 dtype 快路径直接返回 `self`；`empty_like(preserve_format)` 复刻
  ATen `to()` 的 layout 决策，kernel 只做 cast。
- 连续侧用扁平 kernel；非连续侧 `_collapse` 后按 rank 1–4 的 strided
  kernel（`%`/`//` 非 affine 索引）。
- **硬约束 #11**: 所有 Triton 启动包
  `torch_device_fn.device(self.device)` 上下文（否则首次后静默 no-op）。
- 本算子无 `tl.sum` 归约，**#15a 尾块归约坑不适用**（masked load/store 安全）。
- **不用 `@triton.autotune`**（#15b）。
- MLU 后端限制: 非 dense 的「窄→宽」（fp16/bf16→fp32）gather load +
  widening store 在当前 Triton/MLU codegen 会崩，该组合用
  `out.copy_(self)` 兜底；dense/affine 路径不受影响。
- int32 偏移溢出时同样回退。

### 3.3 注册与分发

[kernel/register.py 的 register_a1](../register.py)：`torch.library.Library("aten","IMPL").impl("type_as", fn, "PrivateUse1")`。
Library 对象保持全局引用；调用计数用于框架层跨进程验证（pid 分片）。

### 3.4 三级实现状态

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py) |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py) |
| 硬件级 | ⬜ 置空 | [kernel/hardware_level/README.md](../kernel/hardware_level/README.md)（无厂商 cast 原语） |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 5 shape × 3×3 dtype 组合 + 5 非 dense 视图 + 别名/回退；最差 err **0**；哨兵确定性 ✓ 敏感 ✓ |
| 框架层 | ✅ | `aten::type_as@PrivateUse1` 注册，拦截计数 > 0，拦截路径精度位级一致 |
| 应用层 | ✅ | 子进程混合精度前向命中 1 次，基线与注册输出**逐位一致**（vLLM 未安装，等价 torch API 验证） |
| 黄金回归 | ✅ | `gen_golden` 111 组，reference/torch/triton 均 **111/111** |

### 4.1 kernel 层明细

| shape | dtype | vs 参考 max_err | 哨兵 | 性能(8192²) |
|---|---|---|---|---|
| 64×1024 / 1×14336 / 8×5000 / 128×5120 / 7×4097 | bf16/fp16/fp32 交叉 | 0.0 | 确定性+敏感 | 0.189ms / 2126 GB/s |
| 转置 / permute / 步长 / expand / channels_last | bf16/fp16/fp32 | 0.0（stride 亦一致） | — | — |

### 4.2 framework 层明细

- 调用计数: 应用子进程命中 **1** 次（`A1_TYPE_AS_COUNT_FILE` pid 分片）
- 输出比对: 基线与插件 **逐位一致**（max abs diff = 0）
- 说明: 本机无 vLLM，见 [test/framework_level.py](../test/framework_level.py) 顶部

## 5. 性能

8192×8192，fp16→fp32，warmup 20 + iters 100 + `torch.mlu.synchronize`：

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| **自研 Triton** | 0.190 ms | 2122 GB/s | 1.00x |
| PyTorch 原生 `.to` | **0.175 ms** | 2300 GB/s | 1.09x 更快 |
| FlagGems `ops.to_copy` | 0.317 ms | 1270 GB/s | 0.60x |
| reference（同原生） | 0.176 ms | 2287 GB/s | — |

> 修复了 `common/perf.py` 的 `_sync`：原实现只同步 CUDA，在 MLU 上
> 不同步导致出现 25 TB/s 的假数据；现补 `torch.mlu.synchronize`。

## 6. 已知问题与风险

| # | 问题 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| 1 | 非 dense 窄→宽 gather+widening store 在 MLU codegen 崩溃 | 该组合不走 Triton | `out.copy_(self)` 兜底（结果与原生一致） | 已知，待上游修复 |
| 2 | 自研 Triton 比原生 `.to` 慢 ~9% | 性能 | cast 是内存带宽瓶颈，当前已接近带宽 | 可优化 |
| 3 | vLLM/transformers 未安装 | 应用层非真实推理 | 子进程等价 torch API 验证；vLLM 到位后切换 harness | 环境限制 |
| 4 | MLU 栈要求先 import torch | 导入顺序 | 已修 `common/device.py` | 已缓解 |

## 7. 结论与后续

### 结论

`type_as` 三层全绿、111/111 黄金通过、精度位级一致、哨兵通过、
应用层输出逐位一致。满足 [Must 清单](../../../docs/acceptance.md#levels)；
性能略慢于原生 9%，明显快于 FlagGems `to_copy`。

### 后续工作

- [ ] 安装 vLLM/transformers 后把应用层切到真实 harness
- [ ] 优化连续 cast（更大 BLOCK / 去 mask / 向量化），逼近原生 0.175ms
- [ ] 非 dense 窄→宽用两阶段 kernel 替掉 `out.copy_` 兜底

## 附录: 复现命令

```bash
cd ops/type_as
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.type_as
python3 scripts/perf_compare.py --device cambricon
```
