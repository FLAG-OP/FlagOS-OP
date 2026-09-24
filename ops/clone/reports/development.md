# 算子开发报告: clone

| 项 | 值 |
|---|---|
| 算子名称 | `clone` |
| 实现路线 / 开发级别 | **A1** torch 算子替换 / **Triton 级**（+ torch 级对照） |
| 目标设备 | `cambricon`（Cambricon MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 1. 环境配置

| 项 | 值 |
|---|---|
| 硬件 | Cambricon MLU590-M9 ×8（单卡 98304 MiB），firmware v1.5.0 / CNMON v6.2.29 |
| PyTorch / torch_mlu | 2.7.1+cpu / 1.29.2+torch2.7.1 |
| Triton / FlagGems | 3.2.0+mlu1.7.2 / 5.3.5 |
| dispatch key | `PrivateUse1`（`mlu`） |
| vLLM / transformers | 未安装 |

详见 [configs/devices/cambricon.yaml](../../../configs/devices/cambricon.yaml)。
`python3 scripts/check_env.py --device cambricon` → OK。

## 2. 算子定义

### 2.1 语义

`aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor`：
返回独立新张量，shape/dtype/device 同 self、绝不共享存储；layout 默认
`preserve_format`（dense 保留 stride，否则转 dense）。

```
out = empty_like(self, memory_format=...)   # 新的存储
复制 self 的全部元素到 out
```

### 2.2 参考与接口

见 [reference.py](../reference.py)（原生 `self.clone`）。
Triton 级: `clone_triton(src, memory_format)`。

### 2.3 数值规格

支持 bf16/fp16/fp32；纯拷贝无算术，精度位级一致；容差 fp32 1e-5 /
fp16,bf16 1e-2；shape 含非整倍数 14336/5000/5120/4097。

## 3. 实现说明

- 路线: 标准 aten 算子 → A1（[集成指南](../../../docs/integration.md)）。
- kernel: [kernel/triton_level.py](../kernel/triton_level.py)。`empty_like`
  决定 layout，连续侧扁平 kernel、非连续侧 `_collapse` 后 rank 1–4
  strided kernel。
- **#11**: 启动包 `torch_device_fn.device`。无归约，不涉 #15a；不用 autotune。
- int32 偏移溢出回退 `out.copy_(src)`。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py)（empty_like + copy_） |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py) |
| 硬件级 | ⬜ 置空 | [kernel/hardware_level/README.md](../kernel/hardware_level/README.md) |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 5 shape × 3 dtype + 5 view；最差 err 0；存储独立 ✓；哨兵 ✓ |
| 框架层 | ✅ | `aten::clone@PrivateUse1`，拦截 6 次，数值/存储独立一致 |
| 应用层 | ✅ | 子进程命中 1 次，基线与注册输出逐位一致 |
| 黄金回归 | ✅ | 48/48（reference/torch/triton） |

## 5. 性能

8192² fp32，warmup 20 + iters 100 + MLU synchronize：

| 实现 | 延迟 | 带宽 |
|---|---|---|
| 自研 Triton | 0.242 ms | 2214 GB/s |
| torch 级 | 0.238 ms | 2256 GB/s |
| reference（原生 clone） | **0.238 ms** | 2257 GB/s |

FlagGems 无 `clone` 算子，无第三方可比。自研与原生基本持平（差 ~2%）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 未做 VMM/页级优化 | 与原生持平，无额外收益 | 可接受 |
| 2 | vLLM 未安装 | 应用层为等价验证 | 环境限制 |

## 7. 结论与后续

三层全绿，48/48 黄金，精度位级一致、存储独立，性能与原生持平。
后续: 安装 vLLM 后切真实 harness；如需可尝试大块向量化进一步逼近带宽峰值。

## 附录: 复现命令

```bash
cd ops/clone
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```
