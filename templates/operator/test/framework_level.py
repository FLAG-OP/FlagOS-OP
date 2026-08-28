# 应用层验证骨架: 真实 vLLM 基线/插件双跑（复用本库 harness）。
#
# 不要自己拉起 vLLM——直接复用 tests/framework_level/_harness.py:
#   - 输入/tokenize/黄金比对/计数 已就绪
#   - 注入机制见 docs/testing.md#framework 与三条路线文档
from __future__ import annotations


def run(profile):
    # <TODO: 组装注入环境>
    #   A1: sitecustomize 桥 + FLAGGEMS 黑名单
    #   A2/B: VLLM_FL_PLUGIN_MODULES=<本模块> + VLLM_FL_PER_OP 钉选
    # 然后调用 harness 的基线/插件双跑，断言:
    #   1) 调用计数 > 0（pid 分片汇总）
    #   2) 输出比对（恒等实现→前缀全等；自定义数值→前 2 token 一致率）
    #   3) 黄金回归
    return {"ok": True, "calls": "<TODO>", "output_match": "<TODO>"}
