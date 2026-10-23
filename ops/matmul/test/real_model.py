#!/usr/bin/env python3
"""Compare baseline and cross-process A1 Triton on a tiny real vLLM model."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from common.device import load_profile
    from tests.framework_level._aten_harness import ensure_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor-parallel-size", type=int,
                    help="exploratory override; default uses the P800 profile")
    args = ap.parse_args()
    profile = load_profile("p800-kunlunxin")
    tensor_parallel_size = (args.tensor_parallel_size if args.tensor_parallel_size is not None
                            else int(profile.framework["tensor_parallel_size"]))
    if tensor_parallel_size < 1:
        ap.error("--tensor-parallel-size must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "matmul" / "real_model" / f"{run_id}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"real vLLM logs: {run_dir}", flush=True)
    temp_dir = ROOT / "results" / "t"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    model = ensure_model()
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT)
    base_env["VLLM_LOGGING_LEVEL"] = "WARNING"
    base_env["HF_HOME"] = str(run_dir / "hf_cache")
    excluded = set(filter(None, base_env.get("VLLM_FL_FLAGOS_BLACKLIST", "").split(",")))
    # These two FlagGems ops fail before model generation on this locked P800 stack.
    base_env["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(sorted(excluded | {"copy_", "lt"}))
    runner = HERE / "test" / "real_model_runner.py"

    def run(name: str, env: dict, marker: Path | None = None):
        out = run_dir / f"{name}.json"
        command = [sys.executable, str(runner), "--model", model,
                   "--out", str(out), "--tensor-parallel-size",
                   str(tensor_parallel_size)]
        if marker is not None:
            command.extend(("--marker", str(marker)))
        with (run_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if code:
                raise subprocess.CalledProcessError(code, command)
        return json.loads(out.read_text())

    try:
        baseline = run("baseline", base_env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"baseline failed: {exc}; see {run_dir / 'baseline.log'}")
        return 1
    injected_env = base_env.copy()
    injected_env["PYTHONPATH"] = str(HERE / "injection") + os.pathsep + str(ROOT)
    injected_env.update({
        "FLAGOS_MATMUL_A1_INJECT": "1",
        "FLAGOS_MATMUL_ROOT": str(ROOT),
        "FLAGOS_MATMUL_PROFILE": "p800-kunlunxin",
        "FLAGOS_MATMUL_HIT_LOG": str(run_dir / "hits.log"),
        "FLAGOS_MATMUL_HIT_MARKER": str(run_dir / "generating"),
        "FLAGOS_MATMUL_BOOT_LOG": str(run_dir / "bootstrap.log"),
    })
    try:
        injected = run("injected", injected_env, run_dir / "generating")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"injected run failed: {exc}; see {run_dir / 'injected.log'}")
        return 1
    hits = (run_dir / "hits.log")
    count = len(hits.read_text().splitlines()) if hits.exists() else 0
    match = baseline == injected
    print(f"real vLLM: output_match={match} matmul_hits={count} logs={run_dir}")
    return 0 if match and count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
