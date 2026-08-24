# FlagOS 算子开发模板库 · 文档

## 阅读路径

- 入门: [getting-started.md](getting-started.md) — 5 分钟跑通第一个样例
- 体系: [architecture.md](architecture.md) — 三条路线 × 两级测试的设计
- 路线详解:
  - [route-a1-aten.md](route-a1-aten.md) — Triton → torch aten dispatcher
  - [route-a2-dispatch.md](route-a2-dispatch.md) — Triton → FlagOS dispatch
  - [route-b-vendor.md](route-b-vendor.md) — 厂商语言 → vendor backend
- 测试: [testing.md](testing.md) — 两级测试 / 黄金输出 / 漂移实验 / 输入模板
- 泛化: [device-profiles.md](device-profiles.md) — 芯片泛化与新芯片接入
- 排障: [known-issues.md](known-issues.md) — 本机已知问题清单（9 条实测）

## 快速链接

| 需求 | 入口 |
|---|---|
| 跑全部 6 格矩阵 | `./scripts/run_all.sh` |
| 看某格的可运行样例 | `examples/` |
| 加一种新芯片 | `docs/device-profiles.md` |
| 排查数值问题 | `docs/known-issues.md` |
| 校准断言前缀 | `scripts/drift_study.py` |
