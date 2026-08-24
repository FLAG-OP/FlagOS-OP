#!/usr/bin/env python3
"""样例 A1×框架层: aten 覆盖注入真实 vLLM 前向。

框架级三要素（vLLM v1 前向跑在 EngineCore 子进程，主进程注册不传播）:
  ① sitecustomize 注入——每个子进程启动时注册 aten::silu
  ② FlagGems 黑名单——防止 flag_gems.enable() 覆盖我们的注册
  ③ PER_OP 钉路径——迫使模型走 F.silu → aten::silu 命中计数

运行: python3 examples/a1-framework/example.py [设备profile名]（约 2 分钟）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.framework_level import test_a1

    profile = load_profile(
        sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[a1-framework 样例] 设备: {profile.summary()}")
    print("  注入: sitecustomize(子进程注册) + FlagGems黑名单 + PER_OP钉reference")
    print("  断言: 调用计数>0 / 前缀恒等(自适应) / 黄金共识回归\n")

    ok = test_a1.run(profile)
    assert ok
    print("\n=> A1×framework 样例 PASS")
    print("   同内容正式测试: python3 run.py --route a1 --level framework")


if __name__ == "__main__":
    main()
