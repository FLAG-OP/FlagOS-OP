# C2 真实引擎验证记录: embedding × vLLM-Ascend（ascend910）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-30（复跑；首测 2026-10-29，结论一致） |
| profile | `ascend910`（Ascend 910 / torch_npu 2.10.0，16 卡可见） |
| 引擎 | **vLLM 0.20.2** + **vllm-ascend**（`/vllm-workspace/vllm-ascend` 源码树，`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（本地 snapshot，bf16，max_model_len=192） |
| harness | `ops/embedding/script/c2_real_vllm_ascend.py` |
| consumer_type | `real-engine`（vLLM-Ascend 推理引擎，非 mini consumer） |
| 证据等级 | **C2**（真实引擎命中目标算子） |
| 产物 | [c2_ascend910/summary.json](c2_ascend910/summary.json) 及 baseline/inject 明细 |
| commit | 见 summary.json `commit` 字段 |
| **C2 结论** | **c2-hit**（真实前向命中 `aten::embedding`） |

## 1. 命令（可复现）

```bash
cd /root/FlagOS-OP
python3 ops/embedding/script/c2_real_vllm_ascend.py \
  --out ops/embedding/reports/c2_ascend910
```

环境要点（脚本已内置）：

- `PYTHONPATH=/vllm-workspace/vllm-ascend` + CANN python site-packages（含 `acl`）
- `VLLM_PLUGINS=ascend`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`（规避 NPU fork 再初始化）
- `FLAGOS_A1_ATEN_INJECT=1` + `FLAGOS_A1_OPS=embedding` + `A1_DISPATCH_KEY=PrivateUse1`
- 模型路径可用 `FLAGOS_MODEL_PATH` 覆盖

## 2. 双跑结果

| 跑 | ok | token_ids（贪心 8 token） | text | init_s | wall_s |
|---|---|---|---|---|---|
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 24.05 | 40.73 |
| inject（A1 embedding） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 32.57 | 62.41 |

- **outputs_equal = true**（token_ids 逐位一致，`common_prefix_len=8/8`）
- 注入后输出与基线一致是预期行为：embedding 前向是索引查找，
  FlagOS 实现与 torch_npu 原生实现逐位等价（黄金测试 20/20 PASS）。
- 两次独立复跑（10-29、10-30）结论一致：c2-hit、hits=9、输出逐位一致。

## 3. 调用计数（按 pid 角色，防串算）

本次复跑三份计数文件（`embedding_hits_<pid>.json`），角色由 harness 按
`pid == 主进程` / `pid == EngineCore 子进程` 标注：

| pid | 角色 | hits | 说明 |
|---:|---|---:|---|
| 918474 | main | 0 | 主进程不做模型前向（`forward_hits_before_probe=0`），符合预期 |
| 921889 | engine/worker | 0 | 同引擎侧的非前向子进程（tokenizer/辅助），不命中 |
| 921890 | engine/worker | **9** | **EngineCore：真实前向命中 `aten::embedding`** |

| 计数点 | hits | 说明 |
|---|---:|---|
| 真实前向（`forward_hits_before_probe`） | **0** | 主进程不执行模型前向（值在 inject.json 主进程侧） |
| engine_core_hits（按 pid 角色汇总，仅 EngineCore） | **9** | 见上表 921890 |
| 主进程 / tokenizer 子进程 | 0 | 预期：仅 EngineCore 执行模型前向 |

**9 次的分解**（交叉核对 `inject.stdout.log` 引擎日志）：

- **1 次**：引擎初始化的 profile/dummy 前向——`profile_run` 走
  `_dummy_run(..., skip_attn=True)`（vllm `v1/worker/gpu/model_runner.py:544`），
  attention 被跳过但 `embed_tokens` 照常执行 → 只给 embedding 计数；
- **8 次**：generate 的 8 次前向 = **1 prefill + 7 decode**
  （`max_tokens=8`，首 token 由 prefill 产出）；
- 同批 sdpa_math 运行的 attention 计数为 240 = 30 层 × 8 次 generate
  前向（init dummy 因 `attn_metadata=None` 提前返回、不命中），
  两条证据互相印证前向次数。

- `a1_counter_retrieved = true`：sitecustomize 注册链在 EngineCore 内生效。
- **C2 成立**：真实 vLLM-Ascend 前向命中 A1 拦截目标，非 probe 伪命中
  （主进程 probe 命中为 0，未混入引擎计数）。

## 4. dispatch key 选择与 vllm_ascend 冲突规避

首次尝试用 `AutogradPrivateUse1` 注册时被拒绝：

```text
RuntimeError: This is not allowed since there's already a kernel
registered from python overriding embedding's behavior for
AutogradPrivateUse1 dispatch key and aten namespace.
```

根因：`vllm_ascend` 在 import 时已注册 `aten::embedding` 的
`AutogradPrivateUse1` kernel。推理（no_grad）路径 dispatch 到
`PrivateUse1`（非 autograd key），因此：

- harness 显式设 `A1_DISPATCH_KEY=PrivateUse1`；
- embedding `register.py` 的 `_backend_for` 已映射
  `PrivateUse1 → _ascend`，无需改动 kernel。

修复后：`PrivateUse1` 注册成功，EngineCore 前向命中 9 次，
token_ids 与基线一致。

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
| embedding 真实引擎验证 | **C2 成立**：真实前向命中 9 次，双跑完成，输出与基线逐位一致 |
| sdpa_math Ascend | 不在本记录范围——见 `ops/sdpa_math/reports/c2_ascend910.md` |
| FlagOS/插件版本、模型、profile、commit、注入方式、调用计数 | §1–§5 已记录 |
| 性能门禁 | 双跑 wall_s 仅作可跑性参考；注入 vs 基线延迟对比待后续门禁工作 |
