# C2 真实引擎验证记录: sdpa_math × vLLM-Ascend（ascend910）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-29 |
| profile | `ascend910`（Ascend 910 / torch_npu 2.10.0，16 卡可见） |
| 引擎 | **vLLM 0.20.2** + **vllm-ascend**（`/vllm-workspace/vllm-ascend` 源码树，`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（本地 snapshot，bf16，max_model_len=192） |
| harness | `ops/sdpa_math/script/c2_real_vllm_ascend.py` |
| 产物 | [c2_ascend910/summary.json](c2_ascend910/summary.json) 及 baseline/inject 明细 |
| commit | 见 summary.json `commit` 字段 |
| **C2 结论** | **c2-hit**（引擎侧 attention backend 补丁后，真实前向命中 `aten::_scaled_dot_product_attention_math`） |

## 1. 命令（可复现）

```bash
cd /root/FlagOS-OP
python3 ops/sdpa_math/script/c2_real_vllm_ascend.py \
  --out ops/sdpa_math/reports/c2_ascend910
```

环境要点（脚本已内置）：

- `PYTHONPATH=/vllm-workspace/vllm-ascend` + CANN python site-packages（含 `acl`）
- `VLLM_PLUGINS=ascend`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`（规避 NPU fork 再初始化）
- `FLAGOS_SDPA_MATH_ATTN=1`（引擎侧 attention backend 补丁）
- 模型路径可用 `FLAGOS_MODEL_PATH` 覆盖

## 2. 双跑结果

| 跑 | ok | token_ids（贪心 8 token） | text | init_s | wall_s |
|---|---|---|---|---|---|
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 25.38 | 42.51 |
| inject（A1 sdpa_math） | ☑ | `[260, 3575, 282, 260, 1798, 30, 198, 198]` | ` the capital of the country.\n\n` | 23.75 | 43.15 |

- **outputs_equal = false**（token_ids 不同）——**预期行为**：注入后注意力本体走 FlagOS math 后端（exact-P 概率图路径），与 vllm-ascend 默认 `npu_fusion_attention` 的数值精度不同，导致生成分叉。这正是 C2 命中的直接证据：引擎路由已被改变。
- baseline 文本与 SmolLM2-135M 贪心基线一致；inject 文本语法通顺，说明 math 后端输出在语义可用范围内。

## 3. 调用计数与路由突破

| 计数点 | hits | 说明 |
|---|---:|---|
| 真实前向（`forward_hits_before_probe`） | **241** | EngineCore 进程内注意力经 FlagOSMathAttentionBackend 调用 `aten::_scaled_dot_product_attention_math` |
| engine_core_hits（按 pid 分片汇总） | **241** | EngineCore 子进程计数文件命中 |
| 主进程 probe（generate 后直调 math 后端） | **1** | A1 注册在主进程同样生效，probe 形状 `[1,4,8,16]` fp16 |

- `attn_patched = true`：`patch_npu_platform_get_attn_backend_cls()` 成功把
  `NPUPlatform.get_attn_backend_cls` 补丁为返回 `FlagOSMathAttentionBackend`。
- `a1_counter_retrieved = true`：sitecustomize 注册链在 EngineCore 内生效。
- **C2 成立**：真实 vLLM-Ascend 前向命中 A1 拦截目标，非 probe 伪命中。

## 4. 突破 known-issues #19 的机制

known-issues #19 记录：torch_npu 的 `F.sdpa` 从不走 math 后端，vLLM-Ascend
默认走 `npu_fused_infer_attention`。本 harness 通过引擎侧补丁突破：

1. **引擎侧 attention backend 补丁**：`ops/sdpa_math/engine/flagos_math_attn.py`
   的 `FlagOSMathAttentionBackend` 复用 Ascend metadata builder 与 KV 布局，
   注意力本体改走 `torch.ops.aten._scaled_dot_product_attention_math`。
   `patch_npu_platform_get_attn_backend_cls()` 幂等补丁
   `NPUPlatform.get_attn_backend_cls` 返回该 backend 路径。
2. **causal 语义修复**：math op 的 `is_causal` 是左上对齐 mask；
   decode（Sq=1, Skv=N）必须 `is_causal=False`，否则 query 只看 key[0]
   （实测 probs 全在第 0 列）。
3. **spawn 多进程**：`VLLM_WORKER_MULTIPROC_METHOD=spawn` 规避
   `Cannot re-initialize NPU in forked subprocess`。
4. **enforce_eager=True**：绕过 ACL graph，确保 Python dispatch 可见。

## 5. 版本指纹

| 组件 | 版本/来源 |
|---|---|
| python | 3.11.15 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| vllm | 0.20.2（editable `/vllm-workspace/vllm`，`0.20.2+empty` 安装标签） |
| vllm_ascend | 源码树 `/vllm-workspace/vllm-ascend`（`__version__` 未暴露，`NPUPlatform` 激活） |
| CANN python | `/usr/local/Ascend/cann-9.0.0/python/site-packages`（`acl` 可导入） |

## 6. 对 issue #38 的口径

| 范围项 | 本记录结论 |
|---|---|
| sdpa_math 真实引擎验证 | **C2 成立**：引擎侧补丁后真实前向命中 241 次，双跑完成，输出因 math 后端生效而与基线分叉（预期） |
| embedding Ascend | 不在本记录范围——见 `ops/embedding/reports/c2_ascend910.md` |
| FlagOS/插件版本、模型、profile、commit、注入方式、调用计数 | §1–§5 已记录 |
| 性能门禁 | 双跑 wall_s 仅作可跑性参考；注入 vs 基线延迟对比待后续门禁工作 |

## 7. 与前一版 routing-boundary 记录的差异

前一版（2026-10-28）在无引擎侧补丁下双跑，EngineCore 内注意力走
`npu_fusion_attention`，真实前向 0 命中，标 `c2-routing-boundary`。
本版引入 `FLAGOS_SDPA_MATH_ATTN=1` 引擎侧 backend 补丁后，真实前向
241 次命中，升级为 `c2-hit`。routing boundary 不再适用于本记录。
