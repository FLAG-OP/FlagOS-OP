"""应用层：多 dtype Transformer 激活在隔离子进程中消费 A1 注册。"""
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

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _app_forward(device: str, dtype_name: str):
    """模拟 Transformer 门控激活：bias + sigmoid + reduction。"""
    import torch
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(20261026)
    # [batch, sequence, hidden]，覆盖常见 LLM 激活布局。
    x = torch.randn(2, 64, 4096, generator=generator,
                    dtype=torch.float32).to(device=device, dtype=dtype)
    bias = torch.randn(4096, generator=generator, dtype=torch.float32)
    bias = bias.to(device=device, dtype=dtype) * 0.02
    gated = torch.sigmoid(x + bias)
    return gated.float().mean(dim=-1)


def _child(out_path: str, register: bool, device: str,
           dispatch_key: str, platform: str, dtype_name: str) -> None:
    import torch
    if register:
        import importlib
        reg = importlib.import_module("ops.sigmoid.register")
        reg.register_a1(dispatch_key, platform=platform)
    y = _app_forward(device, dtype_name)
    calls = 0
    if register:
        calls = reg.CALL_COUNT["sigmoid"]
    torch.save(y.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps({
        "calls": calls, "dtype": dtype_name,
        "shape": [2, 64, 4096],
    }), encoding="utf-8")


def _run_child(register: bool, profile, dtype_name: str):
    import torch
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "result.json")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", out,
               "--torch-device", profile.torch_device,
               "--dispatch-key", profile.dispatch_key,
               "--platform", profile.name,
               "--dtype", dtype_name]
        if register:
            cmd.append("--register")
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=os.environ.copy())
        meta = json.loads(Path(out).read_text(encoding="utf-8"))
        tensor = torch.load(out + ".pt", weights_only=True)
    return meta, tensor


def run(profile):
    import torch
    results = {}
    total_calls = 0
    for dtype_name, atol in (("float16", 1e-4),
                             ("bfloat16", 1e-4),
                             ("float32", 2e-6)):
        _, baseline = _run_child(False, profile, dtype_name)
        plug_meta, plugged = _run_child(True, profile, dtype_name)
        assert plug_meta["calls"] > 0, \
            f"{dtype_name} 应用前向未消费 aten::sigmoid 注册"
        torch.testing.assert_close(plugged, baseline, atol=atol, rtol=atol)
        results[dtype_name] = {
            "calls": plug_meta["calls"], "output_match": True,
        }
        total_calls += plug_meta["calls"]

    return {
        "ok": True,
        "engine": "subprocess-transformer-activation",
        "application_shape": [2, 64, 4096],
        "dtypes": results,
        "calls": total_calls,
    }


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child(sys.argv[sys.argv.index("--child") + 1],
               "--register" in sys.argv,
               sys.argv[sys.argv.index("--torch-device") + 1],
               sys.argv[sys.argv.index("--dispatch-key") + 1],
               sys.argv[sys.argv.index("--platform") + 1],
               sys.argv[sys.argv.index("--dtype") + 1])
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "p800-kunlunxin")))
