# aten.sigmoid（昆仑芯 P800）

本目录实现标准 `aten::sigmoid(Tensor self) -> Tensor` 推理前向。交付路线为 A1：通过 `torch.library` 注册到 `CUDA` dispatch key，在 P800 XMLIR 栈上调用 Triton/FlagGems `pointwise_dynamic` kernel。

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

## 支持范围

| 维度 | 已验证范围 |
|---|---|
| 设备 | Kunlunxin P800，官方 FlagGems 5.0 软件锁 |
| dtype | 自研：float16、bfloat16、float32；原生回退：int32、int64、bool |
| shape | 0D～4D、空张量、边界尺寸、长尾和 Transformer 激活 |
| layout | contiguous、转置、步长切片、expand |
| 集成 | A1 注册；C1 mini Transformer 激活消费 |
| 范围外 | sigmoid_、sigmoid.out、sigmoid_backward、硬件级 XTC、C2 真实模型 |

非 strided 或不在自研范围的 dtype 通过 ATen redispatch 保持原生行为。完整范围见 [reports/scope.md](reports/scope.md)。

## 实现结构

- `reference.py`：fp32 中间精度语义参考。
- `kernel/torch_level.py`：ATen 组合实现。
- `kernel/triton_level.py`：可移植 Triton 入口与原生回退。
- `kernel/backends/kunlunxin.py`：P800 `pointwise_dynamic` 与 CodeGenConfig。
- `register.py`：A1 注册、平台校验与调用计数。
- `test/`：kernel、op、C1 应用三层测试。
- `goldendata/`：19 个 shape、3 个核心 dtype、93 份黄金数据。
- `script/`：黄金、三方性能矩阵和 dispatch 测试。

## 复现

本服务器把物理 XPU 2 映射为容器逻辑 `cuda:0`，使用不提交的本地 profile `p800-liyanz-local`。标准双卡环境可改用 `p800-kunlunxin-gems5`。

```bash
python scripts/check_env.py --device p800-liyanz-local
python ops/sigmoid/script/gen_golden.py --device cpu
python ops/sigmoid/script/check_accuracy.py \
  --impl triton --device p800-liyanz-local
python ops/sigmoid/example.py p800-liyanz-local
python ops/sigmoid/script/bench_matrix.py \
  --profile p800-liyanz-local --warmup 20 --iters 100
python ops/sigmoid/script/bench_dispatch.py --device p800-liyanz-local
```

## 当前结论

- kernel 75 例、golden 93/93、A1 9 次命中、C1 三种 dtype 均通过。
- 正式 5.0 性能矩阵 36/36 完成；当前自研 12 个组合均慢于原生，最佳为 `2048×2048 bf16` 的 `1.158×`。
- A1 注册相对直调增加 `7.908 μs`（16.5%）。
- 该版本完成的是功能、精度、应用范围和性能基线建设，不宣称已经获得普遍加速。

## 交付自查

- [x] reference、三级状态、A1 注册和包化入口齐备。
- [x] 三层测试、黄金数据和 dtype/shape/layout/special 矩阵通过。
- [x] 自研、FlagGems、原生三方性能矩阵与 dispatch 开销完成。
- [x] catalog、生成索引和中央性能注册表已更新。
- [x] 开发、测试、精度、性能和应用范围报告已填写。
- 不适用：本次只交付 P800 单平台，因此不创建多平台 `MERGE.md`；硬件级、backward 和 C2 明确属于范围外。
