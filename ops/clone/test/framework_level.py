# 应用层验证: 真实推理式的混合精度前向，跨进程验证 aten 注册被消费。
#
# 本机未安装 vLLM，故用「模板式 + 等价 torch API」子进程验证（见
# ops/type_as/test/framework_level.py 的同一约定）: 子进程=应用进程模型，
# 基线/注册双跑，断言调用计数 > 0 且输出逐位一致。
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
    """模拟应用的混合精度前向: clone 一份激活再 matmul。"""
    import torch

    torch.manual_seed(20260919)
    n, k = 1024, 256
    act = torch.randn(n, k, dtype=torch.float16, device=device)
    weight = torch.randn(k, k, dtype=torch.float16, device=device) * 0.02
    h = act.clone()                         # ← 被测 aten::clone
    return h @ weight.t()


def _app_child(out_path: str, register: bool, device: str) -> None:
    import torch

    if register:
        import register as reg
        reg.register_a1("PrivateUse1")
    y = _app_forward(device)
    calls = None
    if register:
        import register as reg
        calls = reg.CALL_COUNT["clone"]
    torch.save(y.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps({
        "sum": float(y.float().sum().item()), "calls": calls}))


def _run_child(register: bool, device: str):
    import torch

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "child.json")
        env = os.environ.copy()
        env["A1_CLONE_COUNT_FILE"] = os.path.join(d, "counts.json")
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
    assert calls > 0, "应用层未命中 clone（aten 注册未被消费）"
    output_match = bool(base_t.shape == plug_t.shape
                        and (base_t - plug_t).abs().max() == 0)
    assert output_match, \
        f"注册前后应用输出不一致: sum {base_meta['sum']} vs {plug_meta['sum']}"

    return {"ok": True,
            "engine": ("vllm" if VLLM_AVAILABLE else
                       "subprocess-app-fallback (vLLM 未安装)"),
            "calls": calls, "output_match": output_match}


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        j = sys.argv.index("--device")
        _app_child(sys.argv[i + 1], "--register" in sys.argv, sys.argv[j + 1])
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "cambricon")))
