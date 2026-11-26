# scatter_ 交付报告（Ascend910）

2026-11-26，Daizijian：Triton 级 / A1，profile `ascend910`。
覆盖无归约 src/value 重载；推理用途。src 已补齐**真实 vLLM C2**；value 为 C1。
参考 P800 gems5 PR #62：kernelGen 生成的 TLE/DSA 初稿在本栈
`TritonToLinalgIncubated` 编译失败，故改写为纯 Triton（语义不变）。

| 检查 | 实测结果 |
|---|---|
| kernel 直测 | 41/41 |
| 额外边界 | 11/11（越界拒绝、重复索引、4D/5D、bool/int 等） |
| A1 | 41/41，原地返回 + `_version` 计数正确 |
| mini consumer | token-selection 输出精确一致，命中 1 |
| 真实 vLLM C2 | 随机权重 Qwen3；src 生成阶段命中 8 次，两请求 16 token 与基线精确一致 |
| 黄金 | CPU 生成 + Ascend Triton 回放 41/41，SHA256 校验 |
| framework | `vllm.LLM.generate`（PlatformFL / npu），基线/注入同配置 |
| 单测 | `pytest tests/unit` 96 passed |

## 平台要点

- A1 key：`AutogradPrivateUse1`（+PrivateUse1）；`scatter_.src` / `value`。
- 纯 Triton：host 端计算 dst/src 偏移；BLOCK=1024；Python 侧越界检查。
- fp64 回退原生 redispatch（BiSheng 不支持 64-bit float）。
- `_version` 由实现递增（Autograd key 下 dispatcher 不自增）。

## 复现

见 [reports/vllm.md](reports/vllm.md) 与 [reports/ascend910.md](reports/ascend910.md)；
P800 参考材料见 [reports/p800-reference/](reports/p800-reference/)。
