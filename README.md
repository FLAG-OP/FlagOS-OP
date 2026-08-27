# FlagOS-OP

FlagOS 自定义算子开发与验证模板库。

三条实现路线（怎么接入）× 三层验证（在哪验证）构成 9 格矩阵，
芯片差异通过设备 profile 接入。

```
应用层    vLLM · transformers           ← framework 验证
  ↓
框架层    PyTorch + FlagOS 融合算子     ← op 验证 · A1/A2 路线
  ↓
编译层    Triton → 芯片编译栈           ← Triton 级开发
  ↓
算子库层  FlagGems · 厂商 kernel        ← kernel 验证 · B 路线
  ↓
硬件层    XPU · GPU · NPU              ← 设备 profile
```

| | 算子库层 · kernel 直测 | 框架层 · op 注册/分发 | 应用层 · framework 验证 |
|---|---|---|---|
| **A1** torch 算子替换 | Triton kernel 直测 | aten 注册+拦截 | aten 恒等计数注入 |
| **A2** FlagOS 融合算子 | Triton kernel 直测 | dispatch 注册+策略切换 | vendor 身份注入 |
| **B** 厂商算子注册 | 厂商 kernel 直测 + 哨兵检查 | vendor 注册/选择 | audit vendor 拦截 |

## 快速开始

```bash
DEVICE=<profile名> ./scripts/run_all.sh     # 全矩阵 + 一致性
python3 run.py --route a1 --level kernel    # 单格
python3 run.py --list                       # 查看可用设备
python3 examples/b-fullstack/example.py     # 全链路样例（约 3 分钟）
```

## 文档

完整文档在 [docs/](docs/index.md)，建议阅读顺序:

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [快速开始](docs/getting-started.md) | 5 分钟跑通第一格 |
| 2 | [体系结构](docs/architecture.md) | 物理栈 · 开发级别 · 路线 · 3×3 矩阵 |
| 3 | [全链路指南](docs/fullstack-guide.md) | 一个算子从源码到推理的完整流程 |
| 4 | [测试体系](docs/testing.md) | 验证层级 · 黄金 · 漂移 · 一致性 |
| 5 | [已知问题](docs/known-issues.md) | 11 条实测记录 + 检测方法 |

深入主题: [A1](docs/route-a1-aten.md) / [A2](docs/route-a2-dispatch.md) / [B](docs/route-b-vendor.md) 路线详解 ·
[设备接入](docs/device-profiles.md) · [报告](docs/reporting.md) ·
[样例索引](examples/README.md)（13 个可运行样例 + 定位图）

## 许可

[Apache-2.0](LICENSE) · [贡献指南](CONTRIBUTING.md) · [更新日志](CHANGELOG.md)
