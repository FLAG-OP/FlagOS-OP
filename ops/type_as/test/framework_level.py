# 应用层验证: 真实推理式的混合精度前向，跨进程验证 aten 注册被消费。
#
# 现状: 本机未安装 vLLM/transformers，无法拉起官方 vLLM harness
# （tests/framework_level/_harness.py 的 runner 依赖 vllm，且针对
# silu_and_mul 钉选）。故本文件退化为「模板式 + 本机可跑的等价验证」:
#   - 子进程 = 应用的进程模型（vLLM v1 前向也在子进程，主进程注册不传播）
#   - 子进程内用与 vLLM 相同的 torch API（.type_as）走混合精度前向
#   - 基线 / 注册双跑，断言「调用计数 > 0」且「输出与基线逐位一致」
# vLLM 到位后，把 _app_child 换成官方 harness 即可，接口不变。
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

VLLM_AVAILABLE = False
try:  # 仅为报告注明引擎可用性
    import importlib.util
    VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
except Exception:  # noqa: BLE001
    VLLM_AVAILABLE = False


def _app_forward(device: str):
    """模拟应用的混合精度前向: 激活 cast 到权重 dtype 再 matmul。

    这是 transformers/vLLM 里真实存在的调用模式（`.type_as` 被应用代码
    直接消费），而不是测试自造的一层 wrapper。
    """
    import torch

    torch.manual_seed(20260918)
    n, k = 1024, 256
    act = torch.randn(n, k, dtype=torch.float32, device=device)
    weight = torch.randn(k, k, dtype=torch.float16, device=device) * 0.02
    h = act.type_as(weight)                 # ← 被测 aten::type_as
    y = h @ weight.t()
    return y


def _app_child(out_path: str, register: bool, device: str) -> None:
    import torch

    if register:
        import register as reg
        reg.register_a1("PrivateUse1")

    y = _app_forward(device)

    calls = None
    if register:
        import register as reg
        calls = reg.CALL_COUNT["type_as"]

    torch.save(y.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps({
        "sum": float(y.float().sum().item()),
        "calls": calls,
    }))


def _run_child(register: bool, device: str) -> tuple[dict, "object"]:
    import torch

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "child.json")
        env = os.environ.copy()
        env["A1_TYPE_AS_COUNT_FILE"] = os.path.join(d, "counts.json")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--child", out, "--device", device]
        if register:
            cmd.append("--register")
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        meta = json.loads(Path(out).read_text())
        tensor = torch.load(out + ".pt")
    return meta, tensor


def run(profile):
    dev = profile.torch_device
    base_meta, base_t = _run_child(register=False, device=dev)
    plug_meta, plug_t = _run_child(register=True, device=dev)

    calls = plug_meta.get("calls") or 0
    assert calls > 0, "应用层未命中 type_as（aten 注册未被消费）"
    output_match = bool(base_t.shape == plug_t.shape
                        and (base_t.float() - plug_t.float()).abs().max() == 0)
    assert output_match, \
        f"注册前后应用输出不一致: sum {base_meta['sum']} vs {plug_meta['sum']}"

    return {"ok": True,
            "engine": ("vllm" if VLLM_AVAILABLE else
                       "subprocess-app-fallback (vLLM 未安装)"),
            "calls": calls,
            "output_match": output_match,
            "detail": (f"{calls} 次命中；基线与注册输出逐位一致"
                       + ("" if VLLM_AVAILABLE else
                          "；应用层用等价的 torch API 子进程验证，"
                          "非真实 vLLM 推理"))}


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        out_path = sys.argv[i + 1]
        j = sys.argv.index("--device")
        _app_child(out_path, "--register" in sys.argv, sys.argv[j + 1])
    else:
        from common.device import load_profile
        name = sys.argv[1] if len(sys.argv) > 1 else "cambricon"
        print(run(load_profile(name)))
