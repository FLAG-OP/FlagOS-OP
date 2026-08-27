"""intake manifest 契约校验器的纯逻辑单测（无需 torch/GPU）。"""
import json

import pytest

from conftest import _load_script

iv = _load_script("intake_validate")


def good_manifest(**over):
    m = {
        "schema": 1, "name": "case-x", "source": "kernelgen",
        "op": "gelu_and_mul",
        "kernel": {"module": "kernel.py", "entry": "gelu_and_mul"},
        "inputs": [{"name": "x", "shape": [64, 128],
                    "dtype": "bfloat16", "scale": 2.0}],
        "tolerance": {"max_abs_err": 0.01},
        "targets": {"routes": ["a2"], "levels": ["kernel"]},
    }
    m.update(over)
    return m


def test_good_manifest(tmp_path):
    (tmp_path / "kernel.py").write_text("def gelu_and_mul(x): return x\n")
    assert iv.validate_manifest(good_manifest(), tmp_path) == []


@pytest.mark.parametrize("over,frag", [
    ({"schema": 2}, "schema"),
    ({"source": "openai"}, "source"),
    ({"expect": "maybe"}, "expect"),
    ({"kernel": {"module": "nope.py", "entry": "f"}}, "不存在"),
    ({"op": "unknown_op"}, "语义参考"),
    ({"inputs": [{"name": "x", "shape": [64], "dtype": "int8"}]}, "dtype"),
    ({"tolerance": {"max_abs_err": 0}}, "正数"),
    ({"targets": {"routes": ["x9"], "levels": ["kernel"]}}, "routes"),
    ({"targets": {"routes": ["a1"], "levels": ["op"]}}, "注册信息"),
])
def test_bad_manifests(tmp_path, over, frag):
    (tmp_path / "kernel.py").write_text("def f(): pass\n")
    errs = iv.validate_manifest(good_manifest(**over), tmp_path)
    assert errs and any(frag in e for e in errs), errs


def test_real_intake_cases_validate():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "intake" / "cases"
    n = 0
    for d in root.iterdir():
        if not (d / "manifest.json").is_file():
            continue
        errs = iv.validate_manifest(
            json.loads((d / "manifest.json").read_text()), d)
        assert errs == [], f"{d.name}: {errs}"
        n += 1
    assert n >= 2
