#!/usr/bin/env python3
"""C1/C2 证据门禁（issue #41）。

检查四件事（任何一条失败即非零退出）：

1. `ops/catalog.yaml` 每个算子必须声明 `evidence: C0|C1|C2`；
   `C1`/`C2` 要求 `ops/<dir>/test/framework_level.py` 存在。
2. `C2` 必须给出 `evidence_basis`（一句范围诚实的说明）与
   `evidence_paths`（仓库内文件列表），且每个路径真实存在——
   证据声明必须锁到具体文件，防止只写结论不给产物。
3. 平台状态三方一致：`ops/catalog.yaml` 的 platform_families 状态
   ↔ `docs/operator-catalog.md` 状态表 ↔ 生成的 `ops/README.md`
   行（README 的两份生成口径由 `build_ops_index.py --check` 兜底，
   本脚本复核状态列）。
4. 日期字段（catalog `updated`、环境锁 `collected_at` /
   `hardware_collected_at`）必须是合法 `YYYY-MM-DD` 且不晚于今天。

用法::

    python3 scripts/evidence_check.py [--root PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

import yaml

VALID_EVIDENCE = ("C0", "C1", "C2")
FAMILY_STATUS_CN = {"partial": "部分交付", "planned": "规划中"}
_GUIDE_TABLE_ROW = re.compile(
    r"^\|\s*`(?P<key>[a-z0-9_]+)`\s*\|\s*(?P<display>[^|]+?)\s*"
    r"\|\s*(?P<status>partial|planned)\b"
)


def _parse_date(value) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        return None


def _check_dates(root: Path, catalog: dict, errors: list[str]) -> None:
    today = _dt.date.today()
    updated = _parse_date(catalog.get("updated"))
    if updated is None:
        errors.append(f"ops/catalog.yaml: updated 非法: {catalog.get('updated')!r}")
    elif updated > today:
        errors.append(f"ops/catalog.yaml: updated 在未来: {updated}")

    env_dir = root / "configs" / "env"
    if env_dir.is_dir():
        for lock_path in sorted(env_dir.glob("*.lock.yaml")):
            try:
                lock = yaml.safe_load(lock_path.read_text()) or {}
            except yaml.YAMLError as exc:
                errors.append(f"{lock_path}: YAML 解析失败 {exc}")
                continue
            for key in ("collected_at", "hardware_collected_at"):
                if key not in lock:
                    continue
                day = _parse_date(lock.get(key))
                if day is None:
                    errors.append(
                        f"{lock_path.name}: {key} 非法: {lock.get(key)!r}")
                elif day > today:
                    errors.append(
                        f"{lock_path.name}: {key} 在未来: {day}")


def _check_evidence(root: Path, catalog: dict, errors: list[str]) -> None:
    for op in catalog.get("operators", []):
        name = op.get("name", "?")
        level = op.get("evidence")
        if level not in VALID_EVIDENCE:
            errors.append(
                f"operator {name}: evidence 必须是 {VALID_EVIDENCE}，"
                f"当前 {level!r}")
            continue
        directory = op.get("directory", name)
        framework = root / "ops" / directory / "test" / "framework_level.py"
        if level in ("C1", "C2") and not framework.is_file():
            errors.append(
                f"operator {name}: evidence={level} 但缺少 "
                f"ops/{directory}/test/framework_level.py")
        if level == "C2":
            basis = op.get("evidence_basis")
            if not isinstance(basis, str) or not basis.strip():
                errors.append(f"operator {name}: C2 必须提供 evidence_basis")
            paths = op.get("evidence_paths")
            if not isinstance(paths, list) or not paths:
                errors.append(f"operator {name}: C2 必须提供 evidence_paths")
                continue
            for rel in paths:
                if not isinstance(rel, str) or not rel:
                    errors.append(f"operator {name}: evidence_paths 含非法项 {rel!r}")
                    continue
                if rel.startswith(("http://", "https://")):
                    errors.append(
                        f"operator {name}: evidence_paths 必须是仓库内路径，"
                        f"不允许外链 {rel}")
                    continue
                if not (root / rel).exists():
                    errors.append(
                        f"operator {name}: evidence_paths 不存在: {rel}")


def _guide_family_statuses(root: Path, errors: list[str]) -> dict[str, str]:
    guide = root / "docs" / "operator-catalog.md"
    found: dict[str, str] = {}
    if not guide.is_file():
        errors.append("docs/operator-catalog.md 缺失")
        return found
    in_table = False
    for line in guide.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Family key"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            m = _GUIDE_TABLE_ROW.match(line)
            if m:
                found[m.group("key")] = m.group("status")
    if not found:
        errors.append("docs/operator-catalog.md: 未找到家族状态表")
    return found


def _check_family_consistency(root: Path, catalog: dict,
                              errors: list[str]) -> None:
    families = catalog.get("platform_families", {})
    guide = _guide_family_statuses(root, errors)
    for key, meta in families.items():
        want = meta.get("status")
        got = guide.get(key)
        if got is None:
            errors.append(f"docs/operator-catalog.md: 状态表缺少家族 {key}")
        elif got != want:
            errors.append(
                f"平台状态不一致 {key}: catalog={want} "
                f"operator-catalog.md={got}")
    for key in guide:
        if key not in families:
            errors.append(
                f"docs/operator-catalog.md: 状态表含 catalog 没有的家族 {key}")

    readme = root / "ops" / "README.md"
    if not readme.is_file():
        errors.append("ops/README.md 缺失（先运行 scripts/build_ops_index.py）")
        return
    text = readme.read_text(encoding="utf-8")
    for key, meta in families.items():
        status_cn = FAMILY_STATUS_CN.get(meta.get("status"))
        if status_cn is None:
            continue  # 非法状态由 build_ops_index 再报
        row = f"| {meta.get('display_name', key)}（`{key}`） | {status_cn} |"
        if row not in text:
            errors.append(f"ops/README.md 缺少状态行: {row}")


def check(root: Path) -> list[str]:
    errors: list[str] = []
    catalog_path = root / "ops" / "catalog.yaml"
    if not catalog_path.is_file():
        return [f"{catalog_path} 缺失"]
    try:
        catalog = yaml.safe_load(catalog_path.read_text())
    except yaml.YAMLError as exc:
        return [f"ops/catalog.yaml: YAML 解析失败 {exc}"]
    if not isinstance(catalog, dict):
        return ["ops/catalog.yaml: 顶层不是映射"]

    _check_dates(root, catalog, errors)
    _check_evidence(root, catalog, errors)
    _check_family_consistency(root, catalog, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录（默认本脚本上级）",
    )
    args = ap.parse_args(argv)
    errors = check(args.root)
    if errors:
        print(f"evidence_check: {len(errors)} 个错误")
        for err in errors:
            print(f" - {err}")
        return 1
    print("evidence_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
