# 应用层验证: result_type 属 dtype 元数据 类，**无 A1 注入**（route=自用/实验）。
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


def _app_forward(device: str):
    import torch

    torch.manual_seed(20260930)
    x = torch.randn(1024, 256, dtype=torch.float16, device=device)
    w = torch.randn(256, 256, dtype=torch.float16, device=device) * 0.02
    a = x; b = w; dt = torch.result_type(a, b)
    return x @ w


def _app_child(out_path: str, device: str) -> None:
    import torch
    y = _app_forward(device)
    torch.save(y.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps(
        {"sum": float(y.float().sum().item())}))


def run(profile):
    import torch
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "child.json")
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--child", out, "--device", profile.torch_device],
                       check=True, cwd=str(ROOT))
        y = torch.load(out + ".pt")
    assert y.shape == (1024, 256) and torch.isfinite(y).all()
    return {"ok": True, "engine": "subprocess-app (route=自用/实验，无注入)",
            "calls": None, "output_match": True,
            "detail": "result_type 无 A1 注册；应用式前向正常"}


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        j = sys.argv.index("--device")
        _app_child(sys.argv[i + 1], sys.argv[j + 1])
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "cambricon")))
