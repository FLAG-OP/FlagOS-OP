# thead-ppu-c2 embedding C2 报告（真实 vLLM EngineCore）

日期：2026-10-28 · 提交线索：wt-2026-10-26-fix  
profile：`thead-ppu-c2`（真实引擎基座，与 dev 基座 `thead-ppu` 双 profile）

## 1. 环境

| 项 | 值 |
|---|---|
| 容器 | `flagos-ppu-c2`（镜像 `flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012`） |
| Torch | 2.11.0+v0.1.0.ppu2.1.0.oe |
| Triton | pip 3.5.0+gitcc5446cf / import 3.6.0（见 lock notes） |
| vLLM | 0.24.0+empty（editable `/opt/vllm`） |
| vllm-plugin-FL | 0.3.0rc2.post3（editable `/opt/vllm-plugin-FL`，FL 插件激活） |
| FlagGems | 5.4.0+g0d95c2092 |
| 设备 | PPU-ZW810E × 16；C2 用卡 **10,14**（实测 10MiB/0% 空闲） |
| 模型 | tiny Llama `/tmp/flagos_tiny_llama`（`_aten_harness.ensure_model()` 生成，LlamaForCausalLM，hidden 64、2 层） |

环境核对：

```bash
python3 scripts/check_env.py --device thead-ppu-c2
```

## 2. C2 结论（状态：PASS）

真实 vLLM `LLM.generate`（EngineCore 子进程）默认路径经 `F.embedding`
→ `aten::embedding`。sitecustomize 注入 A1 后，EngineCore 内命中 **10** 次，
生成 token 与无注入基线**逐位一致**。

| 断言 | 结果 |
|---|---|
| `embedding_hits_gt_0` | ✅ 10 |
| `output_tokens_match` | ✅ `[[21,1,18,10,1,18,14,1]]` |

## 3. 方法

双跑分进程（A1 注册不可撤销）：

1. **baseline**：真实 vLLM，无注入  
2. **injected**：`FLAGOS_C2_EMB_INJECT=1` + `A1_DISPATCH_KEY=CUDA`  
   + `A1_EMB_PLATFORM=thead-ppu` + `A1_EMBEDDING_COUNT_FILE`（pid 分片 JSON）

入口：

```bash
CUDA_VISIBLE_DEVICES=10,14 \
  python3 ops/embedding/script/e2e_vllm_c2.py \
  --model /tmp/flagos_tiny_llama \
  --out ops/embedding/reports/e2e_vllm_c2_thead-ppu-c2.json
```

## 4. 证据

| 文件 | 内容 |
|---|---|
| [e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json) | 双跑 token / hits / 断言 / status=PASS |
| `ops/embedding/script/e2e_vllm_c2.py` | C2 双跑脚本 |
| `injection/sitecustomize.py`（`FLAGOS_C2_EMB_INJECT`） | EngineCore 内 embedding A1 注入 |

## 5. 备注

- 默认 vLLM 注意力走 flagos/FA3，**不**消费 math 算子（与 sdpa_math C2 相关）。
- embedding 是 decoder 默认必经路径，无需显式改注意力。
- 卡况随时间变化；复跑前用 `ppu-smi` 确认空闲卡，必要时改脚本 `CUDA_VISIBLE_DEVICES`。
