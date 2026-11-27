# aten.sigmoid（昆仑芯 P800）

本目录实现标准 `aten::sigmoid(Tensor self) -> Tensor` 推理前向。交付路线为 A1：通过 `torch.library` 注册到 `CUDA` dispatch key，在 P800 XMLIR 栈上调用 Triton/FlagGems `pointwise_dynamic` kernel。

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

> 2026-11-27 当前状态：连续 `numel=255` 的无 mask 专用路径及 CPU
> 回归已经完成；历史硬件结果均在下文标注日期。当前真实 vLLM C2 的
> baseline 在 TP=2 worker 初始化时遇到 `NCCL error: unhandled cuda error`，
> 因而修改后的三层、Triton 黄金、C2 和性能最终复测仍未闭环。

## 支持范围

| 维度 | 已验证范围 |
|---|---|
| 设备 | Kunlunxin P800，官方 FlagGems 5.0 软件锁 |
| dtype | 自研：float16、bfloat16、float32；原生回退：int32、int64、bool |
| shape | 0D～4D、空张量、边界尺寸、长尾和 Transformer 激活 |
| layout | contiguous、转置、步长切片、expand |
| 集成 | A1 注册；C1 mini Transformer；真实 vLLM C2 脚本已具备、当前复测待运行时恢复 |
| 范围外 | sigmoid_、sigmoid.out、sigmoid_backward、硬件级 XTC |

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

复现只使用仓库已提交的 `p800-kunlunxin-gems5`。目标容器映射宿主机
`/dev/xpu1`、`/dev/xpu2`，并设置两个可见设备变量为 `1,2`；XMLIR
逻辑顺序为物理 XPU 1 → `cuda:0`、物理 XPU 2 → `cuda:1`。算子层按
profile 使用 `cuda:1`，真实 vLLM 使用两张可见卡、TP=2 和 `mp` executor。

```bash
export XPU_VISIBLE_DEVICES=1,2
export CUDA_VISIBLE_DEVICES=1,2
python scripts/check_env.py --device p800-kunlunxin-gems5
python ops/sigmoid/script/gen_golden.py --device cpu
python ops/sigmoid/script/check_accuracy.py \
  --impl triton --device p800-kunlunxin-gems5
python ops/sigmoid/example.py p800-kunlunxin-gems5
python ops/sigmoid/script/bench_matrix.py \
  --profile p800-kunlunxin-gems5 --warmup 20 --iters 100
python ops/sigmoid/script/bench_dispatch.py \
  --device p800-kunlunxin-gems5
python ops/sigmoid/script/c2_real_vllm.py \
  --profile p800-kunlunxin-gems5
```

## 当前结论

- 2026-11-09 修改前历史基线：kernel 75 例、golden 93/93、A1 9 次命中、C1 三种 dtype 均通过；这些数字不是当前提交的硬件验收结果。
- 2026-11-09 历史性能矩阵 36/36 完成；自研 12 个组合均慢于原生，最佳为 `2048×2048 bf16` 的 `1.158×`。历史原始 JSON 未入库且旧服务器路径已不可访问，当前只能作为不可独立复核的摘要。
- 历史 dispatch：A1 注册相对直调增加 `7.908 μs`（16.5%）。
- 2026-11-27 当前 PR：197 项 CPU 单测和证据检查通过；CPU 黄金 reference/torch 各 93/93；C2 baseline 在 EngineCore TP worker 初始化阶段因 NCCL 错误退出，尚未进入生成或 sigmoid 注入阶段。
- 当前代码不宣称已完成硬件全量验收，也不宣称已经获得普遍加速。

## 交付自查

- [x] reference、三级状态、A1 注册和包化入口齐备。
- [ ] 修改后的三层、Triton 黄金和 dtype/shape/layout/special 矩阵待 P800 环境恢复后复测（仅 2026-11-09 历史基线通过）。
- [ ] 修改后的三方性能矩阵待复测（历史 36/36 与 dispatch 结果保留）。
- [x] catalog 和生成索引已更新；中央性能入口仅保留历史 `NEW` 记录，不视为回归门禁通过。
- [x] 开发、测试、精度、性能和应用范围报告已填写，并区分历史基线与当前复测状态。
- 不适用：本次只交付 P800 单平台，因此不创建多平台 `MERGE.md`；硬件级和 backward 属于范围外。真实 vLLM C2 属于当前验收项，尚待运行时恢复后复测。
