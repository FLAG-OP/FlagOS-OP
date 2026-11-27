# P800 当前提交非 C2 验收（2026-11-27）

> 本轮结果绑定提交 `8a99186c841f52b9cd2428071619737ef7b564ec` 和容器镜像
> `sha256:e180a0dd4267ebdfbce4ea5434710870a6ea7e96118457538cc02442907d54b1`。
> 每个阶段只执行一次，正确性通过后才采集性能。

## 结果

| 阶段 | 当前结果 | 关键数值 |
|---|---|---|
| 环境锁 | PASS | Python 与 8 个软件包全部匹配 `p800-kunlunxin-gems5` |
| kernel | PASS | 75 例、19 shape；max abs/rel 均为 0；确定性和敏感性通过 |
| A1 op | PASS | `AutogradCUDA+CUDA`；拦截 9 次，契约计数 9 次 |
| C1 | PASS | `2×64×4096`；fp16/bf16/fp32 3/3 输出一致且均命中 |
| Triton 黄金 | PASS | 93/93；最差 abs `1.192e-07`、rel `1.680e-07` |
| 三方性能矩阵 | 已采集 | 36/36；自研/原生 `1.198–5.736×`，0/12 快于原生 |
| A1 dispatch | 已采集 | 直调 `54.655 μs`，注册 `59.369 μs`，增量 `4.714 μs`（8.6%） |
| C2 vLLM TP=2 | 未重跑 | 已由另一组材料确认在 `ncclCommInitRank` 阶段受环境阻塞 |

## 结论

- 当前提交的 kernel、A1 op、C1、黄金精度、dtype/shape/layout/special 覆盖均已在 P800 完成复测。
- 性能数据可复核，但自研实现没有在 12 个组合中超过原生，因此默认生产路由继续使用 native，Triton 仅作为验证路线。
- C2 没有被本轮结果覆盖，也不能标记为通过。对应 TP=2/NCCL 原始材料见
  [../p800-runtime-failure-2026-11-27/ANALYSIS.md](../p800-runtime-failure-2026-11-27/ANALYSIS.md)。
- 完整命令、绑定环境和日志摘要见 [manifest.json](manifest.json)；性能原始数据见
  [bench_matrix.json](bench_matrix.json)。

## 原始日志

- [环境锁](check_env.log)
- [kernel 层](kernel_level.log)
- [A1 op 层](op_level.log)
- [C1 应用层](framework_level.log)
- [Triton 黄金](golden_triton.log)
- [性能矩阵](bench_matrix.log)
- [dispatch](bench_dispatch.log)
