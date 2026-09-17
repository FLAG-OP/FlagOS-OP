# 查询 generate_kernel 的输入 schema
import asyncio
import sys

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

TOKEN = sys.argv[1]
URL = "https://kernelgen.flagos.io/sse/"


async def main():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with streamablehttp_client(URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            for t in tools.tools:
                if t.name == "generate_kernel":
                    import json
                    print(json.dumps(
                        t.inputSchema if hasattr(t, "inputSchema")
                        else t.input_schema, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
