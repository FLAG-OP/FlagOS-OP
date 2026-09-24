# 从 mcp_result.json 提取 KernelGen 官方生成的代码到可运行模块
import json
import re
from pathlib import Path

raw = json.loads(
    Path("/root/sdpatten-op/intake_llm/mcp_result.json").read_text())["raw"]

# raw 是 JSON 字符串（torch_code/triton_code/test_func_code/benchmark_func_code）
try:
    parts = json.loads(raw)
except Exception:
    # 容错: 提取 json 对象
    m = re.search(r"\{.*\}", raw, re.S)
    parts = json.loads(m.group(0))

out_dir = Path("/root/sdpatten-op/intake_llm")
for key in ("torch_code", "triton_code", "test_func_code",
            "benchmark_func_code"):
    code = parts.get(key)
    if code:
        fn = {"torch_code": "kernelgen_torch_ref.py",
              "triton_code": "kernelgen_triton.py",
              "test_func_code": "kernelgen_test_gen.py",
              "benchmark_func_code": "kernelgen_bench_gen.py"}[key]
        (out_dir / fn).write_text(code)
        print(f"{fn}: {len(code)} chars")

# 提取 wrapper 入口签名（triton_code 的非 kernel 部分）
tri = parts.get("triton_code", "")
m = re.search(r"^def (\w+)\(", tri, re.M)
wrappers = re.findall(r"^def (\w+)\(", tri, re.M)
print("triton_code 顶层函数:", wrappers)
