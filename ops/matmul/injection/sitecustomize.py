"""Register matmul A1 in every vLLM Python process during an isolated test."""
from __future__ import annotations

import os
import sys

if os.environ.get("FLAGOS_MATMUL_A1_INJECT") == "1":
    try:
        root = os.environ["FLAGOS_MATMUL_ROOT"]
        if root not in sys.path:
            sys.path.insert(0, root)
        from common.device import load_profile
        from ops.matmul.register import dispatch_key_for_profile, register_a1

        profile = load_profile(os.environ["FLAGOS_MATMUL_PROFILE"])
        _MATMUL_LIB = register_a1(
            dispatch_key_for_profile(profile),
            hit_log=os.environ["FLAGOS_MATMUL_HIT_LOG"],
            hit_marker=os.environ["FLAGOS_MATMUL_HIT_MARKER"],
        )
        with open(os.environ["FLAGOS_MATMUL_BOOT_LOG"], "a",
                  encoding="utf-8") as log:
            log.write(f"{os.getpid()} registered\n")
    except Exception:
        import traceback

        with open(os.environ["FLAGOS_MATMUL_BOOT_LOG"], "a",
                  encoding="utf-8") as log:
            log.write(f"{os.getpid()} registration failed\n")
            traceback.print_exc(file=log)
