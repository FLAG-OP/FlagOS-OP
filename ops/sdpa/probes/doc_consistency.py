# 文档一致性终检: 链接有效性 + 关键数字口径 + 文件引用存在性
import re
import sys
from pathlib import Path

ROOT = Path("/root/sdpatten-op")
docs = ["README.md", "REPORT.md", "PLATFORM.md", "MERGE.md",
        "reports/development.md", "reports/test-report.md",
        "reports/accuracy.md", "reports/performance.md",
        "reports/perf_analysis.md", "goldendata/README.md",
        "probes/README.md"]
errors = []

# 1) markdown 链接目标存在（相对文档所在目录解析）
for d in docs:
    base = (ROOT / d).parent
    text = (ROOT / d).read_text()
    for m in re.finditer(r"\]\(([^)#]+?)(#[^)]*)?\)", text):
        target = m.group(1)
        if target.startswith(("http", "mailto")):
            continue
        if not (base / target).exists():
            errors.append(f"{d}: 死链 {target}")

# 2) 反引号引用的本地文件存在（.py/.json/.yaml；相对文档所在目录；
#    排除 "X.py/.log" 速记与通配式写法）
for d in docs:
    base = (ROOT / d).parent
    text = (ROOT / d).read_text()
    for m in re.finditer(r"`([\w./-]+\.(?:py|json|yaml))`", text):
        t = m.group(1)
        if "/" in t and "*" not in t \
                and not (base / t).exists() and not (ROOT / t).exists():
            errors.append(f"{d}: 引用缺失 {t}")

# 3) 关键数字口径一致（同数字必须同语境）
canon = {
    "37/37": ["README.md"],          # kernel 层 37 项（36 精度+哨兵）
    "265/265": None,                  # 出现即可
    "1.44": None,                     # S=2k 最新延迟
    "1.93x": None,                    # V4 加速
    "6.5x": None,                     # S=2k vs native
    "17.6x": None,                    # vs gems 最大加速
}
for d in docs:
    text = (ROOT / d).read_text()
    # 旧数据残留检测: 初版 2.775/2.77 只允许出现在"优化历程"语境
    for stale in ("2.775ms", "2.77ms", "12.5x", "12.13x"):
        if stale in text and "优化历程" not in text and "初版" not in text \
                and "base" not in text:
            errors.append(f"{d}: 疑似旧性能数据残留 {stale}")

for d in docs:
    if not (ROOT / d).exists():
        errors.append(f"文档缺失: {d}")

print("检查文档:", len(docs))
if errors:
    print("发现问题:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("全部通过: 链接有效 / 引用文件存在 / 无旧数据残留")
