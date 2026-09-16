# 单独复现 native 精度对照失败点
import sys

sys.path.insert(0, "/root/sdpatten-op")
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

sys.path.insert(0, "/root/sdpatten-op/script")
from check_accuracy import _load, IMPLS  # noqa: E402

fn = _load(*IMPLS["native"])
idx = json.loads(Path("/root/sdpatten-op/goldendata/index.json").read_text())
gspec_tols = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}

fails = []
for f in idx["files"]:
    d = torch.load("/root/sdpatten-op/goldendata/" + f["file"])
    ins = [t.to("npu:0") for t in d["inputs"]]
    q, k, v = ins[0], ins[1], ins[2]
    m = ins[3] if len(ins) > 3 else None
    out = fn(q, k, v, m, 0.0, d["causal"], None, d["Hq"] != d["Hkv"])
    ref = d["expected"].to("npu:0")
    diff = (out.float() - ref.float()).abs()
    nan_rows = torch.isnan(ref[..., 0])
    if nan_rows.any():
        diff = diff[~nan_rows.unsqueeze(-1).expand_as(diff)]
    err = diff.max().item() if diff.numel() else 0.0
    tol = gspec_tols[d["dtype"]]
    if err >= tol:
        fails.append((f["file"], d["dtype"], err, tol))

print(f"native FAIL 数: {len(fails)}")
for x in fails[:10]:
    print("  ", x)
