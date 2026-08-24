#!/usr/bin/env python3
"""按 inputs/spec.yaml 生成测试输入。

产物:
  tests/framework_level/inputs/prompts.txt      框架级文本输入
  tests/framework_level/inputs/token_inputs.json 算子级 token 输入

用法:
  python3 scripts/gen_inputs.py                 # 按 spec 生成
  python3 scripts/gen_inputs.py --check         # 校验产物与 spec 一致
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = ROOT / "inputs" / "spec.yaml"
OUT_DIR = ROOT / "tests" / "framework_level" / "inputs"


def render_prompts_txt(prompts: list[str]) -> str:
    lines = ["# 自动生成: scripts/gen_inputs.py (勿手改; spec: inputs/spec.yaml)"]
    lines += prompts
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="仅校验产物与 spec 一致，不写文件")
    args = ap.parse_args()

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    from common.inputs import generate_from_spec
    gen = generate_from_spec(spec)

    prompts_txt = render_prompts_txt(gen["framework"])
    prompts_path = OUT_DIR / "prompts.txt"
    tokens_path = OUT_DIR / "token_inputs.json"

    digest = hashlib.sha256("\n".join(gen["framework"]).encode()).hexdigest()[:16]

    if args.check:
        cur = prompts_path.read_text(encoding="utf-8") if prompts_path.exists() else ""
        ok = cur == prompts_txt
        print(f"prompts.txt {'一致' if ok else '不一致'} "
              f"({len(gen['framework'])} prompts, sha={digest})")
        sys.exit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(prompts_txt, encoding="utf-8")
    tokens_path.write_text(__import__("json").dumps(gen["operator"], indent=2))
    print(f"已生成 {prompts_path} ({len(gen['framework'])} prompts, sha={digest})")
    print(f"已生成 {tokens_path} "
          f"({sum(len(v['prompts']) for v in gen['operator'].values())} token 序列)")


if __name__ == "__main__":
    main()
