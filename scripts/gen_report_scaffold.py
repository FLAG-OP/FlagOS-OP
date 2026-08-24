#!/usr/bin/env python3
"""生成算子开发报告骨架（环境配置 + 验证结果自动填充）。

用法:
  python3 scripts/gen_report_scaffold.py --op my_op --route a2 --device p800-kunlunxin
  # 产出 reports/my_op_p800-kunlunxin_report.md

自动填充:
  - 环境配置章节（调用 env_snapshot.collect）
  - 9 格矩阵中该路线 3 格的状态/耗时（读 results/）
  - 跨层一致性 / 黄金摘要（若已跑）
其余章节保留 TODO 占位由开发者手填。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path_hack = __import__("sys")
sys_path_hack.path.insert(0, str(ROOT))


def _cell_status(device: str, route: str, level: str) -> dict | None:
    p = ROOT / "results" / f"{device}_{route}_{level}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fmt_cell(c: dict | None) -> str:
    if c is None:
        return "⬜ 未跑 | — | <手填>"
    return f"{'✅' if c['status'] == 'PASS' else '❌'} {c['status']} | {c['seconds']}s | <手填>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, help="算子名（如 gelu_and_mul）")
    ap.add_argument("--route", required=True, choices=["a1", "a2", "b"])
    ap.add_argument("--device", default="p800-kunlunxin")
    args = ap.parse_args()

    from env_snapshot import collect

    env_md = collect(args.device)

    cells = {lv: _cell_status(args.device, args.route, lv)
             for lv in ("kernel", "op", "framework")}

    report = f"""# 算子开发报告: {args.op}

| 项 | 值 |
|---|---|
| 算子名称 | `{args.op}` |
| 实现路线 | {args.route.upper()} |
| 目标设备 | `{args.device}` |
| 开发者 | `<手填>` |
| 日期 | `<手填>` |
| 报告状态 | 草稿 |

---

{env_md}
---

## 2. 算子定义 `[手填]`

### 2.1 语义

<一段话 + 公式/伪代码>

### 2.2 PyTorch 参考实现

```python
# <手填>
```

### 2.3 数值规格

<dtype / 内部精度 / 容差>

---

## 3. 实现说明 `[手填]`

- 代码位置: `routes/... / examples/...`
- kernel 要点: <手填>
- 注册与分发: <impl_id / 优先级 / PER_OP 钉选>

---

## 4. 验证结果

| 层级 | 状态 | 耗时 | 关键结论 `[手填]` |
|---|---|---|---|
| kernel 直测 | {_fmt_cell(cells['kernel'])} |
| op 注册/分发 | {_fmt_cell(cells['op'])} |
| framework 真实推理 | {_fmt_cell(cells['framework'])} |

### 4.1 kernel 层明细 `[手填]`

| shape | dtype | vs 参考 max_err | 哨兵 | 性能 |
|---|---|---|---|---|
| | | | | |

### 4.2 framework 层明细 `[手填]`

- 调用计数: <从测试输出抄录>
- 输出比对: <模式 + 结果>

---

## 5. 性能 `[手填]`

| 实现 | 耗时/call | 有效带宽 | 相对参考加速 |
|---|---|---|---|
| 本实现 | | | |
| PyTorch 参考 | | | 1x |

> 基准方法: 短采样（≤100 次），避免分配器池增长失真。

---

## 6. 已知问题与风险 `[手填]`

| # | 问题 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| 1 | | | | |

---

## 7. 结论与后续 `[手填]`

### 结论

<是否达到验收标准>

### 后续

- [ ]

---

## 附录 A: 复现命令

```bash
python3 run.py --route {args.route} --level kernel --device {args.device}
python3 run.py --route {args.route} --level op --device {args.device}
python3 run.py --route {args.route} --level framework --device {args.device}
python3 run.py --consistency --device {args.device}
python3 scripts/report.py --device {args.device}
```
"""
    out = ROOT / "reports" / f"{args.op}_{args.device}_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"报告骨架已生成: {out}")
    print("环境配置与验证状态已自动填充; [手填] 项待补。")


if __name__ == "__main__":
    main()
