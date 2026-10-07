from __future__ import annotations

import importlib
import sys
import types


import pytest


@pytest.mark.parametrize(
    ("module_name",),
    [
        ("ops.sdpa.kernel.backends.p800_kunlunxin",),
        ("ops.sdpa.kernel.backends.mlu590",),
    ],
)
def test_backend_survives_foreign_top_level_reference(module_name, monkeypatch):
    """Guard the multi-operator perf-registry import order.

    Earlier standalone operators can leave their own ``reference`` module in
    ``sys.modules``. The backend must resolve its own package reference first;
    otherwise importing ``sdpa_reference`` from that foreign module fails.
    """
    foreign = types.ModuleType("reference")
    monkeypatch.setitem(sys.modules, "reference", foreign)

    module = importlib.import_module(module_name)
    # Package scanning may have imported this backend before this test. Reload
    # after installing the foreign top-level module so the test always covers
    # the import statements rather than a cached module object.
    module = importlib.reload(module)

    assert module.sdpa_reference.__module__ == "ops.sdpa.reference"
