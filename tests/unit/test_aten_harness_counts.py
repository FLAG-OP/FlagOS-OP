"""A1 共享框架 harness 的计数聚合回归测试。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "tests/framework_level/_aten_harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "framework_aten_harness_counts", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(directory: Path, name: str, payload) -> None:
    (directory / name).write_text(
        json.dumps(payload), encoding="utf-8")


def test_mixed_operator_count_keys_are_aggregated(tmp_path):
    harness = _load_harness()
    _write(tmp_path, "embedding.101", {
        "aten::embedding": 20,
        "runner_pid": 101,
    })
    _write(tmp_path, "sdpa_math.202", {
        "aten::_scaled_dot_product_attention_math": 11,
        "registered_at": 123456,
    })
    _write(tmp_path, "sigmoid.303", {
        "sigmoid": 3,
        "dumped_at_epoch": 123457,
    })

    assert harness._agg_counts(str(tmp_path)) == {
        "embedding": 20,
        "sdpa_math": 11,
        "sigmoid": 3,
    }
    assert harness._count_processes(str(tmp_path)) == [
        {"op": "embedding", "pid": 101, "hits": 20,
         "file": "embedding.101"},
        {"op": "sdpa_math", "pid": 202, "hits": 11,
         "file": "sdpa_math.202"},
        {"op": "sigmoid", "pid": 303, "hits": 3,
         "file": "sigmoid.303"},
    ]


def test_legacy_count_keys_remain_compatible(tmp_path):
    harness = _load_harness()
    _write(tmp_path, "embedding.1", {"embedding": 4})
    _write(tmp_path, "sdpa_math.2", {"sdpa_math": 5})
    _write(tmp_path, "sigmoid.3", {"aten::sigmoid": 6})

    assert harness._agg_counts(str(tmp_path)) == {
        "embedding": 4,
        "sdpa_math": 5,
        "sigmoid": 6,
    }


def test_invalid_counts_and_numeric_metadata_are_not_hits(tmp_path):
    harness = _load_harness()
    _write(tmp_path, "embedding.1", {
        "aten::embedding": True,
        "runner_pid": 999,
    })
    _write(tmp_path, "sdpa_math.2", {
        "aten::_scaled_dot_product_attention_math": -1,
        "registered_at": 123456,
    })
    _write(tmp_path, "sigmoid.3", {
        "sigmoid": "3",
        "dumped_at_epoch": 123457,
    })
    _write(tmp_path, "clone.4", {"clone": 1.5, "pid": 4})
    _write(tmp_path, "item.5", {"item": 2, "pid": 5000})
    _write(tmp_path, "copy_.6", [7])
    (tmp_path / "dropout.7").write_text("{broken", encoding="utf-8")

    assert harness._agg_counts(str(tmp_path)) == {"item": 2}
    assert harness._count_processes(str(tmp_path)) == [
        {"op": "item", "pid": 5, "hits": 2, "file": "item.5"},
    ]
