"""Environment checks must report incomplete metadata as a mismatch."""
# tyf-2026-10-23-fix
# tyf <tyanfeng@mail.ustc.edu.cn>
import sys

from scripts import check_env


def test_missing_version_metadata_is_reported_without_crashing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_env.py", "--device", "p800-kunlunxin"])
    lock = check_env.yaml.safe_load(
        (check_env.ENV_DIR / "p800-kunlunxin.lock.yaml").read_text()
    )
    versions = {check_env.DIST.get(k, k): v for k, v in lock["packages"].items()}
    versions["xmlir"] = None
    monkeypatch.setattr(check_env.md, "version", versions.__getitem__)
    monkeypatch.setattr(check_env.platform, "python_version", lambda: lock["python"])
    assert check_env.main() == 1
    output = capsys.readouterr().out
    assert "版本元数据缺失或不可读" in output
    assert "结论: 1 项不符" in output
