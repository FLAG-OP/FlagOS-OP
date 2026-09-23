# p800-kunlunxin SDPA 移植与验证报告

日期：2026-09-22

## 1. 环境

| 项 | 值 |
|---|---|
| Torch | 2.9.0+cu129 |
| torch_xmlir | XMLIR--bc1b1dc6f-dev+2026032411 |
| Triton | 3.0.0+03c4c9be |
| FlagGems | 4.2.1rc0 |
| 设备 | `CUDA_VISIBLE_DEVICES=1,2`，稳定测试卡 `cuda:1` |

环境核对：

```bash
python3 scripts/check_env.py --device p800-kunlunxin
```

## 2. 实现选择

### 2.1 尝试复用 FlagGems `_kunlunxin`

无 mask / causal 路径可编译运行，但性能明显落后；float mask 路径在
`xpuLaunchKernel("_attn_fwd", ...)` 处失败：

```text
RuntimeError: xpuLaunchKernel(...) -> Operation not permitted(err_code: 1)
```

因此不把 FlagGems Triton 版作为 P800 默认实现，仅保留为 benchmark 对照。

### 2.2 尝试平移 ascend Triton kernel

Ascend kernel 在 XMLIR Triton frontend 处失败：

```text
Rewrite for-op failed. Could not find PtrState returned by the loop.
Unsupported select predicate.
```

根因是 Ascend 版为 causal 循环边界生成 runtime select；XMLIR rewrite
无法处理该形态。该结论不能推断所有 Triton kernel 均不可用，只说明此
Ascend 优化不能零改动平移。

### 2.3 最终实现

- fp16/bf16：直调 `aten::_scaled_dot_product_efficient_attention`。
- bool mask：转换为 additive `-inf` float bias。
- 全遮蔽行：厂商 kernel 返回有限值，输出端恢复 CPU/aten 的 NaN 语义。
- q/k/v 直接反传：forward 按需计算 log-sumexp；无梯度/A1 内层前向保持
  `compute_log_sumexp=False` 快路径。
- 可微 float mask：厂商 backward 报 `bias_requires_grad not supported yet`，
  direct 路径复用 A1 的数学 backward（dq/dk/dv/dmask 均可得）。
- fp32：厂商 kernel 误差约 `2e-4`，超过黄金容差；改走 fp32 ATen 组合。
- XMLIR bmm 缺陷：`S∈(320,640]` 可触发 `bmm_one_loop` 编译失败；将
  该区间 pad 到 768（若另一边更长则保持原长），再用合法 mask 还原。

## 3. A1 注册

| 项 | 结论 |
|---|---|
| dispatch key | `AutogradCUDA` |
| Python schema | dispatcher 可能省略默认参数，注册 wrapper 显式补齐 |
| autograd | 复用统一数学 backward；dq/dk/dv 与原生子进程对照通过 |
| direct autograd | q/k/v 走厂商 backward（按需 log-sumexp）；可微 mask 走 A1 数学 backward |
| auto 路由 | P800 直调 private aten，绕过 `F.sdpa` patch，避免递归 |

## 4. 验证结果

```bash
python3 test/kernel_level.py p800-kunlunxin
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 test/op_level.py p800-kunlunxin
python3 test/framework_level.py p800-kunlunxin
python3 probes/guard_check.py p800-kunlunxin
```

| 项 | 结果 |
|---|---|
| kernel 层 | 37/37；普通 fp32 ≤5e-7，bf16 最大 3.906e-3；确定性与输入敏感通过 |
| 黄金 | 397/397；extreme 相对误差 4.25e-3 |
| op 层 | 拦截 count=2；注册=直调逐位；direct q/k/v 与可微 mask 反向通过；dq 4.883e-4 / dk 2.441e-4 / dv 3.906e-3 |
| 应用层 | 拦截 count=28；logits diff=0；top-1 与贪心续写一致率 1.00 |
| 平台守卫 | CPU 拒绝、XMLIR 元数据、AutogradCUDA 注册均通过 |

应用层使用 bf16：该随机 mini-decoder 的 Linear/LayerNorm fp16 链路在
P800 栈上溢出；bf16 下基线与插件输出均稳定。

## 5. 性能采样

口径：**加速比 = baseline 延时 / ours 延时**，>1 表示 ours 更快。
JSON 同时保留历史 `ours_vs_native` 延时比值（Ascend 报告既有 schema）
并新增 `speedup_vs_native` 常规加速比字段。

计时必须读取一个输出元素（如 `output[0, 0, 0, 0].item()`）强制完成。
实测仅调用 kernel 后丢弃返回值、再 `torch.cuda.synchronize()`，在
XMLIR 栈上可能只计入 launch/异步提交时间，得到数十 TFLOPS 到
9000+ TFLOPS 的虚假结果；该口径已在 benchmark 脚本中修复。

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --warmup 20 --iters 100 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
```

| shape | ours(ms) | F.sdpa(ms) | FlagGems(ms) | 加速比 = F.sdpa/ours |
|---|---:|---:|---:|---:|
| prefill_1k_d64 | 0.1484 | 0.1655 | 0.3120 | **1.115** |
| prefill_1k_d128 | 0.1442 | 0.1574 | 0.3606 | **1.092** |
| prefill_2k_d128 | 0.2584 | 0.2941 | 1.3283 | **1.138** |
| prefill_4k_d128 | 0.6632 | 0.6796 | 4.3990 | **1.025** |
| gqa_1k_d128 | 0.1884 | 0.2045 | 0.5667 | **1.085** |
| decode_d128 | 0.1132 | 0.1270 | 0.1611 | **1.122** |

Benchmark 结束时 FlagGems autotuner 可能打印 XPU cleanup 噪声，
不影响已保存的 JSON 与后续进程。

### A100 参照（FA2 论文协议）

A100 为 FlashAttention-2 论文公开数据按同 FLOPs 换算，非同机复测。
P800 使用 `cross_platform_p800-kunlunxin.json` 的强制完成口径：

| 场景 | P800 ours | A100·FA2 | 延时比 |
|---|---:|---:|---:|
| D64 S=2k | 2.156ms | ~0.785ms | 2.75x 慢 |
| D64 S=8k | 7.040ms | ~2.894ms | 2.43x 慢 |
| D128 S=1k | 0.789ms | ~0.344ms | 2.30x 慢 |
| D128 S=4k | 2.312ms | ~1.195ms | 1.94x 慢 |
| D128 S=8k | 4.404ms | ~2.340ms | 1.88x 慢 |

## 6. 遗留与建议

1. `cuda:0` 上厂商 SDPA 曾挂起，回归默认使用 `cuda:1`；根因属设备 /
   runtime 状态，需要厂商侧确认。
2. P800 尚未交付自研 Triton forward/backward；当前策略是优先复用厂商
    efficient kernel，避免重复造轮子并保证应用可用性。
3. XMLIR `bmm_one_loop` 编译缺陷应反馈给栈维护者；当前 workaround 仅
   激活于已知失败区间。
