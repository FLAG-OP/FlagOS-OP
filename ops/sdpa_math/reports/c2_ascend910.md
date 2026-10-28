# C2 真实引擎验证记录: sdpa_math × vLLM-Ascend（ascend910）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-28 |
| profile | `ascend910`（Ascend 910 / torch_npu 2.10.0，16 卡可见） |
| 引擎 | **vLLM 0.20.2** + **vllm-ascend**（`/vllm-workspace/vllm-ascend` 源码树，`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（本地 snapshot，bf16，max_model_len=192） |
| harness | `ops/sdpa_math/script/c2_real_vllm_ascend.py` |
| 产物 | [c2_ascend910/summary.json](c2_ascend910/summary.json) 及 baseline/inject 明细 |
| commit | 见 summary.json `commit` 字段 |
| **C2 结论** | **routing boundary**（不是完整 C2 命中） |

## 1. 命令（可复现）

```bash
cd /root/FlagOS-OP
python3 ops/sdpa_math/script/c2_real_vllm_ascend.py \
  --out ops/sdpa_math/reports/c2_ascend910
```

环境要点（脚本已内置）：

- `PYTHONPATH=/vllm-workspace/vllm-ascend` + CANN python site-packages（含 `acl`）
- `VLLM_PLUGINS=ascend`
- 模型路径可用 `FLAGOS_MODEL_PATH` 覆盖

## 2. 双跑结果

| 跑 | ok | token_ids（贪心 8 token） | text | init_s | wall_s |
|---|---|---|---|---|---|
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 12.48 | 24.81 |
| inject（A1 sdpa_math） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | 同上 | 16.12 | 37.65 |

- **outputs_equal = true**（token_ids 逐位一致）
- 基线与注入输出一致是预期行为：真实前向未命中 math 后端（见 §3）

## 3. 调用计数与路由边界

| 计数点 | hits | 说明 |
|---|---:|---|
| 真实前向（`forward_hits_before_probe`） | **0** | EngineCore/Worker 内注意力经 `npu_fusion_attention`，不调用 `aten::_scaled_dot_product_attention_math`（known-issues #19） |
| 主进程 probe（generate 后直调 math 后端） | **1** | A1 成对注册（`AutogradPrivateUse1` + `PrivateUse1`）在 NPU 进程内生效，`registered_libs=2` |

因此 **不把默认 vLLM-Ascend 路由声明为 sdpa_math 的 C2 命中**。
sdpa_math 在 ascend910 的证据等级保持：

- **C0** 黄金 / kernel / op 层（既有）
- **C1** mini-decoder / 概率图消费者（既有 `test/framework_level.py`）
- **C2** 真实引擎：本记录证明「双跑可完成、输出不被破坏、注册链可用」，
  但默认路由 **0 命中** → 标记 `c2-routing-boundary`，不是完整端到端交付。

## 4. sitecustomize 全进程注入在 NPU 上的额外边界

实验中把 `FLAGOS_A1_ATEN_INJECT=1` + `injection/` 放进 PYTHONPATH 后，
EngineCore 子进程在 `check_ascend_device_type()` 处报：

```text
RuntimeError: Cannot re-initialize NPU in forked subprocess.
To use NPU with multiprocessing, you must use the 'spawn' start method
```

根因: 父进程在 fork 前已通过 A1 注册路径触发 `torch.npu` 初始化，
vLLM-Ascend EngineCore 默认 fork 模型无法再初始化 NPU。
本 harness 因此改为**主进程 probe**（`LLM.generate` 之后再碰 NPU 张量），
不依赖 sitecustomize 注入 EngineCore。

后续若要在 EngineCore 内做 Ascend A1 注入，需要:

1. 确认 vLLM-Ascend 的 multiprocessing start method（spawn / forkserver）；
2. 或把注册推迟到 worker 内 NPU 初始化之后（插件/图编译钩子，见 PR #34 评审的 engine adapter 方向）。

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
| embedding Ascend | 不在本记录范围——已降级为历史实验（PLATFORM.md / catalog，另一 commit） |
| sdpa_math 真实引擎验证 | **已完成双跑与路由边界记录**；默认路由 0 命中，不标完整 C2 |
| FlagOS/插件版本、模型、profile、commit、注入方式、调用计数 | §1–§5 已记录 |
| 性能门禁 | 双跑 wall_s 仅作可跑性参考；未做注入 vs 基线的延迟门禁（0 命中下无意义） |
