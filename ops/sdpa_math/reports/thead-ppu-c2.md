# thead-ppu-c2 sdpa_math C2 报告（真实 vLLM EngineCore）

日期：2026-10-28 · 提交线索：wt-2026-10-26-fix  
profile：`thead-ppu-c2`（真实引擎基座，与 dev 基座 `thead-ppu` 双 profile）

## 1. 环境

| 项 | 值 |
|---|---|
| 容器 | `flagos-ppu-c2`（镜像 `flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012`） |
| Torch | 2.11.0+v0.1.0.ppu2.1.0.oe |
| vLLM | 0.24.0+empty（editable `/opt/vllm`） |
| vllm-plugin-FL | 0.3.0rc2.post3（attention_backend: flagos→vendor→reference） |
| FlagGems | 5.4.0+g0d95c2092 |
| 设备 | PPU-ZW810E；C2 用卡 **10,14** |
| 模型 | tiny Llama `/tmp/flagos_tiny_llama`；`model_impl=transformers` |

环境核对：

```bash
python3 scripts/check_env.py --device thead-ppu-c2
```

## 2. 路径判定（状态：PASS）

**默认真实 vLLM 注意力不消费** `aten::_scaled_dot_product_attention_math`  
（flag_gems `flash_attn_varlen` / FA3 wheel；`ALL_ATTENTION_FUNCTIONS["vllm"]`
→ vLLM Attention → flagos）。`model_impl=transformers` 仍把
`_attn_implementation` 钉为 `"vllm"`，HF 的 F.sdpa 不会走到 math composite。

因此 C2 采用**显式模型路径**（issue 允许的「显式模型路径」口径）：

```
sitecustomize（FLAGOS_C2_SDPA_MATH_INJECT=1）
  ├─ A1 注册 aten::_scaled_dot_product_attention_math（CUDA 成对 key）
  └─ ALL_ATTENTION_FUNCTIONS["vllm"] ← F.sdpa + SDPBackend.MATH
       → 落到 math composite → A1 counter 拦截
```

## 3. C2 结论

双跑（baseline = 同 MATH 路径无 A1；injected = MATH 路径 + A1）：

| 断言 | 结果 |
|---|---|
| `math_hits_gt_0` | ✅ 20 |
| `output_tokens_match` | ✅ 两组均为 `[[0,15,18,22,12,1,18,22]]` |

（flash 基线 token 不同，属路径差异；C2 对照固定为同 MATH 路径。）

## 4. 方法与证据

```bash
CUDA_VISIBLE_DEVICES=10,14 \
  python3 ops/sdpa_math/script/e2e_vllm_c2.py \
  --model /tmp/flagos_tiny_llama --model-impl transformers \
  --out ops/sdpa_math/reports/e2e_vllm_c2_thead-ppu-c2.json
```

| 文件 | 内容 |
|---|---|
| [e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json) | 双跑 token / hits / 断言 / status=PASS |
| `ops/sdpa_math/script/e2e_vllm_c2.py` | C2 双跑脚本 |
| `injection/sitecustomize.py`（`FLAGOS_C2_SDPA_MATH_INJECT` / `FLAGOS_C2_FORCE_MATH_PATH`） | 注入与 baseline 路径补丁 |

## 5. 备注

- A1 注册用 `SDPA_MATH_IMPL=torch`（ATen 组合）；thead flash/efficient
  kernel 已在 kernel/op/framework 层与黄金 175/175 验证（见 `reports/thead-ppu.md`）。
- C2 证明的是：真实 EngineCore 内 **算子被消费** 且 **A1 拦截后输出与同路径基线一致**。
- 复跑前确认空闲卡；tiny 模型由 harness 生成，权重不在镜像内。
