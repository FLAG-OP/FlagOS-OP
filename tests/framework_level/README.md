# 应用层（framework 验证）测试

验证两件事:
1. **算子真的被调用**: 自定义实现被应用层消费者选中执行；消费者可能是
   真实 vLLM 前向，也可能是设备 profile 明确声明的 mini consumer。
2. **端到端不破坏**: 启用后消费者输出符合预期。mini consumer 结果不能
   代替真实 vLLM 证据，报告必须标注实际消费者类型。

## 统一流程（_harness.py 编排）

- 输入: `inputs/prompts.txt`（每行一个 prompt；`PROMPTS_FILE`/`PROMPTS_LIMIT` 可覆盖/限量）
- 基线跑: 无插件，贪心解码，记录输出 token ids
- 插件跑: 按 route 注入对应实现，同 prompt 同 seed
- 断言: 调用计数 > 0 + 输出比对

## 三条路线的注入方式与断言

| 路线 | 注入 | 输出断言 |
|---|---|---|
| A1 | 数值恒等 aten::silu 覆盖 + `PER_OP=silu_and_mul=reference` | 前 8 token 完全一致（恒等数学） |
| A2 | Triton silu_and_mul 以 `vendor:triton-template` 注册 + PER_OP 钉住 | 前 2 token 一致 ≥2/3（自定义数值允许混沌分叉） |
| B | audit vendor + PER_OP 钉住，基线与插件均走 reference（纯拦截） | 前 8 token 完全一致（数值恒等） |

引擎参数全部来自设备 profile（TP/显存/怪癖），跨芯片无需改代码。

## 黄金输出（golden outputs）回归检测

```bash
# 跨设备黄金锚点（多快照 → 多数票共识前缀）
python3 scripts/build_golden.py --device cpu                 # transformers CPU 权威参考
python3 scripts/build_golden.py --device <加速卡profile>    # 示例: p800-kunlunxin
python3 scripts/compare_golden.py --devices <加速卡profile>,cpu
```

- 黄金 = N 次独立运行的快照集合 + 逐位置多数票共识前缀
- CPU 设备走 HuggingFace transformers 直接推理（最权威语义参考、
  确定性、绕过 vLLM/厂商栈）；GPU/XPU 设备走 vLLM reference 路径
- 没有 vLLM runtime 的 profile（例如 MLU590）使用 mini-decoder；这只证明
  应用层消费、注册和数值链路，真实 vLLM 验证须在有 runtime 的环境补跑
- 保存于 `golden/<device>_golden.json`（含引擎参数指纹、prompts 哈希，应提交入库）
- A1/B 框架级测试自动比对黄金共识前缀（跨会话回归检测），exact
  断言前缀按黄金实测稳定前缀自适应
- `scripts/drift_study.py` 可量化本栈跨进程非确定性，校准断言前缀长度
- 输入统一为 tokenize+词表钳制的 token id（`PROMPTS_AS_TOKENS=1`），
  规避 vendor 栈 embedding 静默越界，保证跨设备输入严格一致

## P800 真实 vLLM C2 实测（2026-10-26）

在自建容器 `daizijian-p800-flagosop`（vllm 0.13.0 + vllm_fl 0.1.0 +
torch_xmlir）上，用现成 **Qwen2.5-0.5B-Instruct** 跑通 8 个可注入算子：

```bash
FLAGOS_DEVICE=p800-kunlunxin \
FLAGOS_MODEL_PATH=/workspace/.cache/models/qwen2.5-0.5b \
USE_FLAGGEMS=0 \
CUDA_VISIBLE_DEVICES=0 \
FLAGOS_A1_OPS=type_as,clone,contiguous,dropout,empty_like,empty_strided,item,_local_scalar_dense \
python -m pytest tests/framework_level/test_a1_aten_ops.py -q -s
```

结果：`1 passed in 72.04s`；真实 vLLM 前向命中
`empty_strided=6299`、`empty_like=5917`、`contiguous=2208`（total=14424），
注入前后输出 token ids 逐位一致。模型来源为
`/data2/evalx-workspace/models/Qwen__Qwen2.5-0.5B-Instruct`，复制到自建容器
workspace 内使用，不占用任何其他人的容器。

限制（均在本机实测复现）：

- `copy_` 必须从 `FLAGOS_A1_OPS` 排除：P800 bf16 兜底会走
  `aten::type_as` / `Tensor.to`，而两者内部重新进入已注册的 `copy_`，
  形成 `copy_ ↔ type_as/to` 递归；需改为不经过 `copy_` 的 cast。
- `USE_FLAGGEMS=0`：FlagGems 4.2 的 P800 `gather` 在 vLLM 采样阶段报
  `xpuLaunchKernel ... Operation not permitted(err_code: 1)`；开启
  FlagGems 时 C2 不通过。
- 更大的 Qwen3-8B 在本栈 EngineCore 初始化阶段挂起（GPU 100%、显存仅
  166MiB），因此当前 C2 证据使用 0.5B 现成模型；8B 需要另查栈/模型兼容性。
