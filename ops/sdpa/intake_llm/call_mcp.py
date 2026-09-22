# KernelGen MCP 客户端（streamable HTTP 传输——/sse/ 端点实为新协议）
import asyncio
import json
import sys

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
URL = "https://kernelgen.flagos.io/sse/"

PARAMS = {
    "kernel_name": "scaled_dot_product_attention",
    "func_desc": (
        "Scaled dot product attention: out = softmax(q @ k^T * scale + "
        "mask) @ v. Supports 4D (B,H,S,D) tensors, causal masking, "
        "optional additive/boolean attention mask, GQA (Hq != Hkv)."),
    "func_type": "attention",
    # 服务端 schema: 三者为逗号分隔字符串（skill 文档示例是 list，
    # 实际 /sse/ 端点 pydantic 校验要求 string）
    "arg_names": "q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa",
    "arg_type": ("torch.Tensor, torch.Tensor, torch.Tensor, "
                 "Optional[torch.Tensor], float, bool, Optional[float], bool"),
    "arg_descs": ("query (B, Hq, Sq, D), "
                  "key (B, Hkv, Skv, D), "
                  "value (B, Hkv, Skv, D), "
                  "optional mask broadcastable to (B,Hq,Sq,Skv); bool=keep float=additive, "
                  "dropout probability (0.0 for inference), "
                  "causal lower-triangular mask requires Sq==Skv, "
                  "scale factor default 1/sqrt(D), "
                  "grouped query attention when Hq != Hkv"),
    "output_arg_desc": "attention output (B, Hq, Sq, D), same dtype as q",
    "flagos_wiki": [
        "Kernel pattern: flash-attention tiled two-pass online softmax",
        "Use Triton for GPU parallelization",
        "Input tensors are contiguous row-major layout (B,H,S,D)",
        "Memory access: K/V streamed by blocks along S_kv, Q tile resident",
        "fp32 accumulator for softmax numerical stability",
        "Causal: skip upper-triangle blocks for efficiency",
        "Tail blocks (S not multiple of BLOCK) need masked loads",
    ],
}


async def main():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with streamablehttp_client(URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await session.call_tool("generate_kernel",
                                          arguments=PARAMS)
            out = []
            for c in res.content:
                if hasattr(c, "text"):
                    out.append(c.text)
            text = "\n".join(out)
            with open("/root/sdpatten-op/intake_llm/mcp_result.json",
                      "w") as f:
                f.write(json.dumps({"raw": text}, ensure_ascii=False,
                                   indent=2))
            print("== 结果长度:", len(text))
            print(text[:4000])


if __name__ == "__main__":
    asyncio.run(main())
