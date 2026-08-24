# 应用框架层测试

验证两件事:
1. **算子真的被调用**: 自定义实现在真实 vLLM 前向中被选中执行
2. **端到端不破坏**: 启用后模型输出符合预期

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
python3 scripts/build_golden.py --device p800-kunlunxin --snapshots 3
python3 scripts/compare_golden.py --devices p800-kunlunxin,cpu
```

- 黄金 = N 次独立运行的快照集合 + 逐位置多数票共识前缀
- CPU 设备走 HuggingFace transformers 直接推理（最权威语义参考、
  确定性、绕过 vLLM/厂商栈）；GPU/XPU 设备走 vLLM reference 路径
- 保存于 `golden/<device>_golden.json`（含引擎参数指纹、prompts 哈希，应提交入库）
- A1/B 框架级测试自动比对黄金共识前缀（跨会话回归检测），exact
  断言前缀按黄金实测稳定前缀自适应
- `scripts/drift_study.py` 可量化本栈跨进程非确定性，校准断言前缀长度
- 输入统一为 tokenize+词表钳制的 token id（`PROMPTS_AS_TOKENS=1`），
  规避 vendor 栈 embedding 静默越界，保证跨设备输入严格一致
