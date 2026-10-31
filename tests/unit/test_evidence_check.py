from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

from scripts.evidence_check import check


def _make_repo(
    root: Path,
    *,
    evidence: str | None = "C2",
    basis: str | None = "demo 引擎双跑 2026-10-29",
    paths: list[str] | None = None,
    with_framework: bool = True,
    updated: str = "2026-10-29",
    guide_status: str = "partial",
    catalog_status: str = "partial",
    readme: bool = True,
) -> Path:
    paths = ["evidence.json"] if paths is None else paths
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "demo_op" / "test").mkdir(parents=True, exist_ok=True)
    (root / "evidence.json").parent.mkdir(parents=True, exist_ok=True)

    op: dict = {
        "name": "demo_op",
        "directory": "demo_op",
        "group": "core",
        "purpose": "demo",
        "key_result": "demo",
        "platforms": {"demo": {"profiles": ["demo-1"]}},
    }
    if evidence is not None:
        op["evidence"] = evidence
        if evidence == "C2":
            if basis is not None:
                op["evidence_basis"] = basis
            if paths is not None:
                op["evidence_paths"] = paths
    catalog = {
        "updated": updated,
        "platform_families": {
            "demo": {
                "display_name": "Demo 芯",
                "status": catalog_status,
                "current_profiles": [],
            },
        },
        "operators": [op],
    }
    (root / "ops" / "catalog.yaml").write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8"
    )
    if with_framework:
        (root / "ops" / "demo_op" / "test" / "framework_level.py").write_text(
            "def test_demo():\n    assert True\n", encoding="utf-8"
        )
    (root / "evidence.json").write_text("{}", encoding="utf-8")
    (root / "docs" / "operator-catalog.md").write_text(
        textwrap.dedent(
            f"""
            # catalog

            | Family key | 显示名 | 当前状态 |
            |---|---|---|
            | `demo` | Demo 芯 | {guide_status} |
            """
        ),
        encoding="utf-8",
    )
    if readme:
        (root / "ops").mkdir(exist_ok=True)
        status_cn = "部分交付" if catalog_status == "partial" else "规划中"
        (root / "ops" / "README.md").write_text(
            f"| Demo 芯（`demo`） | {status_cn} | 1 | demo-1 |\n",
            encoding="utf-8",
        )
    return root


def test_minimal_c2_repo_passes(tmp_path: Path):
    _make_repo(tmp_path)
    assert check(tmp_path) == []


def test_missing_evidence_field_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, evidence=None)
    errors = check(tmp_path)
    assert any("evidence 必须是" in e for e in errors)


def test_c1_requires_framework_level_file(tmp_path: Path):
    _make_repo(tmp_path, evidence="C1", basis=None, paths=None,
               with_framework=False)
    errors = check(tmp_path)
    assert any("framework_level.py" in e for e in errors)


def test_c2_requires_basis_and_paths(tmp_path: Path):
    _make_repo(tmp_path, evidence="C2", basis=None, paths=[])
    errors = check(tmp_path)
    assert any("evidence_basis" in e for e in errors)
    assert any("evidence_paths" in e for e in errors)


def test_c2_rejects_missing_or_remote_paths(tmp_path: Path):
    _make_repo(tmp_path, evidence="C2",
               paths=["missing.json", "https://example.com/e.json"])
    errors = check(tmp_path)
    assert any("不存在: missing.json" in e for e in errors)
    assert any("仓库内路径" in e for e in errors)


def test_future_updated_date_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, updated="2099-01-01")
    errors = check(tmp_path)
    assert any("updated 在未来" in e for e in errors)


def test_family_status_mismatch_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, guide_status="planned", catalog_status="partial")
    errors = check(tmp_path)
    assert any("平台状态不一致 demo" in e for e in errors)


def test_readme_missing_family_row_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, readme=False)
    errors = check(tmp_path)
    assert any("ops/README.md 缺失" in e for e in errors)
