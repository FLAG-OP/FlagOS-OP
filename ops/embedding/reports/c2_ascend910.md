# C2 真实引擎验证记录: embedding × vLLM-Ascend（ascend910）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-29 |
| profile | `ascend910`（Ascend 910 / torch_npu 2.10.0，16 卡可见） |
| 引擎 | **vLLM 0.20.2** + **vllm-ascend**（`/vllm-workspace/vllm-ascend` 源码树，`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（本地 snapshot，bf16，max_model_len=192） |
| harness | `ops/embedding/script/c2_real_vllm_ascend.py` |
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
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 25.91 | 42.82 |
| inject（A1 embedding） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 28.58 | 57.65 |

- **outputs_equal = true**（token_ids 逐位一致）
- 注入后输出与基线一致是预期行为：embedding 前向是索引查找，
  FlagOS 实现与 torch_npu 原生实现逐位等价（黄金测试 20/20 PASS）。

## 3. 调用计数

| 计数点 | hits | 说明 |
|---|---:|---|
| 真实前向（`forward_hits_before_probe`） | **9** | EngineCore 进程内 input embeddings 路径命中 `aten::embedding` |
| engine_core_hits（按 pid 分片汇总） | **9** | EngineCore 子进程（pid 183955）计数文件命中 |
| 主进程 / tokenizer 子进程 | 0 | 预期：仅 EngineCore 执行模型前向 |

- `a1_counter_retrieved = true`：sitecustomize 注册链在 EngineCore 内生效。
- **C2 成立**：真实 vLLM-Ascend 前向命中 A1 拦截目标，非 probe 伪命中。

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
