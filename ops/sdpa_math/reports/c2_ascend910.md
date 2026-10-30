# C2 真实引擎验证记录: sdpa_math × vLLM-Ascend（ascend910）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-30（复跑；首测 2026-10-29，结论一致） |
| profile | `ascend910`（Ascend 910 / torch_npu 2.10.0，16 卡可见） |
| 引擎 | **vLLM 0.20.2** + **vllm-ascend**（`/vllm-workspace/vllm-ascend` 源码树，`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（本地 snapshot，bf16，max_model_len=192） |
| harness | `ops/sdpa_math/script/c2_real_vllm_ascend.py` |
| consumer_type | `real-engine`（vLLM-Ascend 推理引擎，非 mini consumer） |
| 证据等级 | **C2**（真实引擎命中目标算子） |
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
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 22.71 | 40.31 |
| inject（A1 sdpa_math） | ☑ | `[260, 3575, 282, 260, 1798, 30, 198, 198]` | ` the capital of the country.\n\n` | 24.15 | 48.68 |

- **outputs_equal = false**（token_ids 不同，共同前缀 1/8）——**例外已处置**，
  详见下方「输出对比处置」。根因：注入后注意力本体走 FlagOS math 后端
  （exact-P 概率图路径），与 vllm-ascend 默认 `npu_fusion_attention` 的
  数值精度不同，导致贪心生成分叉。这正是 C2 命中的直接证据：引擎路由
  已被改变。
- baseline 文本与 SmolLM2-135M 贪心基线一致；inject 文本语法通顺，
  说明 math 后端输出在语义可用范围内。
- 两次独立复跑（10-29、10-30）结论一致：c2-hit、forward_hits=240、
  `outputs_equal=false` 且分叉 token 序列完全相同（确定性复现）。

## 2.1 输出对比处置（outputs_equal=false 的原因 / 定量 / 复现 / 退出条件）

C2 判据是「真实前向命中目标算子 + 完成 baseline/inject 双跑 + 输出对比
有明确结论」，**不要求 token 逐位相等**（两种 attention 实现精度不同，
逐位相等反而说明路由没被改变）。处置如下：

| 项 | 内容 |
|---|---|
| 原因 | math 后端（fp32 累加、返回完整 P）与 `npu_fusion_attention`（融合 kernel、bf16 累加）数值精度不同 → 贪心第 2 个 token 起分叉（共同前缀 1/8） |
| 定量① token 对比 | baseline `[260,2240,282,7042,30,657,314,260]` vs inject `[260,3575,282,260,1798,30,198,198]`；`common_prefix_len=1/8`；`bitwise_equal=false` |
| 定量② teacher-forced prompt logprob | 同一 prompt 逐 token logprob（`prompt_logprobs=0`）：4 个可比位置，`max_abs_diff=0.046174` nats，`mean_abs_diff=0.020734` nats；逐位置 baseline `[-8.2267,-0.8253,-5.2528,-1.1319]` vs inject `[-8.2267,-0.8488,-5.2412,-1.0857]`（首个位置完全相同） |
| 复现 | `python3 ops/sdpa_math/script/c2_real_vllm_ascend.py --out ops/sdpa_math/reports/c2_ascend910`（两次复跑分叉序列与 logprob 差值一致，确定性） |
| 例外范围 | **仅限**「math 后端 vs npu_fusion_attention 精度差导致的贪心分叉」；不豁免 C1 黄金（175/175，容差内）、不豁免 embedding 的逐位一致要求 |
| 退出条件 | 出现下列任一即本例外失效、须重新评估：① prompt logprob `max_abs_diff > 0.5` nats（超出数值精度差量级）；② inject 文本不可用（语法崩坏/重复循环）；③ C1 黄金或 kernel 层出现超容差失败；④ 换模型/换 batch 后分叉发生在首个 token（说明路由或语义被破坏而非精度差） |

## 3. 调用计数与路由突破（按 pid 角色，防串算）

本次复跑三份计数文件（`sdpa_math_hits_<pid>.json`），角色由 harness 按
`pid == 主进程` / `pid == EngineCore 子进程` 标注：

| pid | 角色 | hits | 说明 |
|---:|---|---:|---|
| 880357 | main | 1 | 主进程真实前向 0 + **generate 后直调 probe 1**（A1 链在主进程生效，probe 形状 `[1,4,8,16]` fp16） |
| 884887 | engine/worker | 0 | 同引擎侧的非前向子进程 |
| 884888 | engine/worker | **240** | **EngineCore：真实前向命中 `aten::_scaled_dot_product_attention_math`** |

| 计数点 | hits | 说明 |
|---|---:|---|
| 真实前向（`forward_hits_before_probe`，主进程侧） | 0 | 模型前向在 EngineCore，主进程仅 probe |
| engine_core_hits（按 pid 角色汇总，仅 EngineCore） | **240** | 见上表 884888 |
| 主进程 probe（generate 后直调 math 后端） | **1** | 独立计数，**不计入** engine 的 240 |

**240 的分解**（交叉核对）：240 = **30 层 × 8 次 generate 前向**
（1 prefill + 7 decode，`max_tokens=8` 首 token 由 prefill 产出）。
引擎初始化的 profile/dummy 前向走 `skip_attn=True` 且
`attn_metadata=None` → 引擎 `forward()` 提前返回，不命中（+0）。
交叉验证：以 `prompt_logprobs` 二次 generate 试跑时 engine 计 **270**
= 240 + 30（第二次 generate 仅 1 次 prefill 前向），与逐次前向 30 次的
分解一致。

> 口径说明（回应前一版 241 的疑问）：旧汇总把主进程 probe 的 1 次
> 加进了 engine 计数（当时子进程未记录 `pid`，无法按角色拆分），
> 故出现「241」。本版按 pid 角色拆分后：**engine 240 + 主进程 probe 1**，
> 两者独立列示，互不混算。

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
| sdpa_math 真实引擎验证 | **C2 成立**：引擎侧补丁后 EngineCore 真实前向命中 **240** 次（30 层 × 1 prefill + 7 decode），主进程 probe 1 独立列示；双跑完成，输出与基线分叉（共同前缀 1/8 + prompt logprob max\|Δ\|=0.046 nats）——例外已按 §2.1 处置 |
| embedding Ascend | 不在本记录范围——见 `ops/embedding/reports/c2_ascend910.md` |
| FlagOS/插件版本、模型、profile、commit、注入方式、调用计数 | §1–§5 已记录 |
| 性能门禁 | 双跑 wall_s 仅作可跑性参考；注入 vs 基线延迟对比待后续门禁工作 |

## 7. 与前一版 routing-boundary 记录的差异

前一版（2026-10-28）在无引擎侧补丁下双跑，EngineCore 内注意力走
`npu_fusion_attention`，真实前向 0 命中，标 `c2-routing-boundary`。
2026-10-29/30 引入 `FLAGOS_SDPA_MATH_ATTN=1` 引擎侧 backend 补丁后，
真实前向 **240** 次命中（另主进程 probe 1），升级为 `c2-hit`。
routing boundary 不再适用于本记录（无补丁时的默认路由仍为
`npu_fusion_attention`，见 known-issues #19）。
计数口径变化：旧版 241 = 240 + probe 1（未按 pid 角色拆分），
本版按角色拆分为 engine 240 / main probe 1。
