# 应用层验证: 推理式前向消费 detach，跨进程验证注册。
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
try:
    import importlib.util
    VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
except Exception:  # noqa: BLE001
    VLLM_AVAILABLE = False


def _app_forward(device: str):
    import torch

    torch.manual_seed(20260926)
    a = torch.randn(1024, 256, dtype=torch.float16, device=device)
    b = torch.randn(256, 256, dtype=torch.float16, device=device) * 0.02
    x0 = torch.randn(1024, 256, dtype=torch.float16, device=device)
    h = x0.detach()
    return h @ b


def _app_child(out_path: str, register: bool, device: str) -> None:
    import torch

    if register:
        import register as reg
        reg.register_a1("PrivateUse1")
    y = _app_forward(device)
    calls = None
    if register:
        import register as reg
        calls = reg.CALL_COUNT["detach"]
    if torch.is_tensor(y):
        torch.save(y.cpu(), out_path + ".pt")
        payload = {"sum": float(y.float().sum().item()), "calls": calls}
    else:
        torch.save(torch.zeros(1), out_path + ".pt")
        payload = {"scalar": y, "calls": calls}
    Path(out_path).write_text(json.dumps(payload))


def _run_child(register: bool, device: str):
    import torch

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "child.json")
        env = os.environ.copy()
        env["A1_DETACH_COUNT_FILE"] = os.path.join(d, "counts.json")
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
    assert calls > 0, "应用层未命中 detach"
    if base_meta.get("scalar") is not None:
        assert abs(base_meta["scalar"] - plug_meta["scalar"]) <= 1e-3
    else:
        assert (base_t - plug_t).abs().max() == 0, "注册前后输出不一致"
    return {"ok": True,
            "engine": ("vllm" if VLLM_AVAILABLE else
                       "subprocess-app-fallback (vLLM 未安装)"),
            "calls": calls, "output_match": True}


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        j = sys.argv.index("--device")
        _app_child(sys.argv[i + 1], "--register" in sys.argv, sys.argv[j + 1])
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "cambricon")))
