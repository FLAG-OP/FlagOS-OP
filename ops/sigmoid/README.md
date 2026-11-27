# aten.sigmoid（昆仑芯 P800）

本目录实现标准 `aten::sigmoid(Tensor self) -> Tensor` 推理前向。交付路线为 A1：通过 `torch.library` 注册到 `CUDA` dispatch key，在 P800 XMLIR 栈上调用 Triton/FlagGems `pointwise_dynamic` kernel。

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

> 2026-11-27 当前状态：连续 `numel=255` 的无 mask 专用路径、200 项 CPU
> 回归、P800 三层测试、Triton 黄金 93/93 和 36 项性能矩阵均已完成。
> 真实 vLLM C2 baseline 仍在 TP=2 worker 的 `ncclCommInitRank` 阶段受
> 环境阻塞，因此 C2 不标记为通过。

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

- 当前 PR：200 项 CPU 单测和证据检查通过；P800 kernel 75 例、A1 9 次命中、C1 三种 dtype、Triton 黄金 93/93 均通过。
- 当前性能矩阵 36/36 已采集，自研/原生为 `1.198–5.736×`，12 个组合均未超过原生；A1 dispatch 增量为 `4.714 μs`（8.6%）。
- C2 baseline 在 EngineCore TP worker 的 `ncclCommInitRank` 阶段因 NCCL 错误退出，尚未进入生成或 sigmoid 注入阶段。
- 当前代码不宣称已完成 C2 验收或获得普遍加速。可复核日志见 [当前验收材料](evidence/p800-current-validation-2026-11-27/ANALYSIS.md) 和 [环境故障材料](evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md)。

## 交付自查

- [x] reference、三级状态、A1 注册和包化入口齐备。
- [x] 修改后的三层、Triton 黄金和 dtype/shape/layout/special 矩阵已在 P800 复测通过。
- [x] 修改后的三方性能矩阵与 dispatch 已采集并提交原始证据；默认路由因无加速保持 native。
- [x] catalog 和生成索引已更新；中央性能入口仅保留历史 `NEW` 记录，不视为回归门禁通过。
- [x] 开发、测试、精度、性能和应用范围报告已填写，并区分历史基线与当前复测状态。
- 不适用：本次只交付 P800 单平台，因此不创建多平台 `MERGE.md`；硬件级和 backward 属于范围外。真实 vLLM C2 属于当前验收项，尚待运行时恢复后复测。
