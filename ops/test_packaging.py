# 包化验证: ops.sdpa + ops.embedding 同进程包态导入 + e2e 双注册
# 运行: cd /root/FlagOS-OP && python3 ops/test_packaging.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

# ── 1. 包态导入不冲突（原病根: 顶层 register/kernel 互相遮蔽）──
import ops.sdpa as sdpa_pkg
import ops.embedding as emb_pkg

assert sdpa_pkg.PLATFORM.startswith("multi"), sdpa_pkg.PLATFORM
from ops.sdpa.register import register_a1 as sdpa_register_a1
from ops.embedding.register import register_a1 as emb_register_a1
print("[1] 双算子包态导入 OK（无名字遮蔽）")

# ── 2. 双注册同进程（原病根: e2e 三缺陷场景）──
import os
os.environ.setdefault("SDPA_DISPATCH_MODE", "auto")
from ops.sdpa.kernel.auto_dispatch import install_patch, stats, reset_stats
install_patch()
emb_register_a1("AutogradPrivateUse1", platform="ascend910")
print("[2] sdpa patch + embedding aten 注册 同进程 OK")

# ── 3. 端到端消费: 两算子都被接管且数值正确 ──
g = torch.Generator().manual_seed(3)
F = torch.nn.functional
reset_stats()

# embedding 消费
w = (torch.randn(1000, 64, generator=g) * 0.1).half().to("npu:0")
idx = torch.randint(0, 1000, (4, 32), generator=g).to("npu:0")
out_emb = F.embedding(idx, w)
ref_emb = w.cpu()[idx.cpu()].to("npu:0")
err_e = (out_emb.float() - ref_emb.float()).abs().max().item()

# sdpa 消费（大 S 走截流原生, 小 S 走 Triton——两分支都过）
qL = torch.randn(1, 4, 4096, 64, generator=g).half().to("npu:0")
kL = torch.randn(1, 4, 4096, 64, generator=g).half().to("npu:0")
vL = torch.randn(1, 4, 4096, 64, generator=g).half().to("npu:0")
oL = F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
qS = torch.randn(1, 4, 256, 64, generator=g).half().to("npu:0")
kS = torch.randn(1, 4, 256, 64, generator=g).half().to("npu:0")
vS = torch.randn(1, 4, 256, 64, generator=g).half().to("npu:0")
oS = F.scaled_dot_product_attention(qS, kS, vS, is_causal=True)

refS = torch.nn.functional.scaled_dot_product_attention  # patched
from ops.sdpa.reference import sdpa_reference
rS = sdpa_reference(qS, kS, vS, None, 0.0, True, None, False)
err_s = (oS.float() - rS.float()).abs().max().item()

assert err_e == 0.0, f"embedding err {err_e}"
assert err_s < 2e-2, f"sdpa 小S err {err_s}"
assert stats["routed_native"] >= 1 and stats["routed_triton"] >= 1, stats
print(f"[3] 双算子消费 OK: embedding err={err_e}, sdpa 小S err={err_s:.2e}")
print(f"    sdpa 路由: native={stats['routed_native']} "
      f"triton={stats['routed_triton']}")

# ── 4. 单算子旧用法兼容（脚本模式 sys.path 注入）──
import subprocess
r = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '/root/FlagOS-OP/ops/embedding')\n"
     "from register import register_a1\n"
     "print('legacy-ok', register_a1.__module__)"],
    capture_output=True, text=True, timeout=300)
assert "legacy-ok" in r.stdout, r.stdout[-200:] + r.stderr[-200:]
print("[4] 单算子旧用法（sys.path 注入）兼容 OK")

print("\nPACKAGING ALL GREEN")
