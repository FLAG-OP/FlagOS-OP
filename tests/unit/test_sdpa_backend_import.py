from __future__ import annotations

import importlib
import sys
import types


def test_p800_backend_survives_foreign_top_level_reference(monkeypatch):
    """Guard the multi-operator perf-registry import order.

    Earlier standalone operators can leave their own ``reference`` module in
    ``sys.modules``. The backend must resolve its own package reference first;
    otherwise importing ``sdpa_reference`` from that foreign module fails.
    """
    foreign = types.ModuleType("reference")
    monkeypatch.setitem(sys.modules, "reference", foreign)

    module = importlib.import_module(
        "ops.sdpa.kernel.backends.p800_kunlunxin"
    )
    module = importlib.reload(module)

    assert module.sdpa_reference.__module__ == "ops.sdpa.reference"
