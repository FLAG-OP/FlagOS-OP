# FlagOS-OP

FlagOS 自定义算子开发与验证模板库。

三条实现路线（怎么接入）× 三层验证（在哪验证）构成 9 格矩阵，
芯片差异通过设备 profile 接入。

```
应用层    vLLM · transformers
           └ 验证: framework 验证（真实推理）
  ↓
框架层    PyTorch + FlagOS dispatch
           ├ 验证: op 验证（注册/分发/拦截）
           └ 路线: A1 torch 算子替换 · A2 FlagOS 融合算子
  ↓
编译层    Triton → 芯片编译栈
           └ 开发: Triton 级（自研设备码）
           └ 旁路: AI 生成源（KernelGen）→ intake 契约验证
  ↓
算子库层  FlagGems · 厂商 kernel
           ├ 验证: kernel 验证（直测+哨兵）
           ├ 路线: B 厂商算子注册
           └ 开发: torch 级 · 硬件级
  ↓
硬件层    XPU · GPU · NPU
           └ 设备: profile 接入
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
python3 scripts/perf_compare.py --device p800-kunlunxin   # 性能回归门禁
python3 scripts/intake_validate.py --validate-only        # AI 生成算子契约
```

## 文档

完整文档在 [docs/](docs/index.md)，建议阅读顺序:

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [快速开始](docs/getting-started.md) | 5 分钟跑通第一格 |
| 2 | [体系结构](docs/architecture.md) | 物理栈 · 开发级别 · 路线 · 3×3 矩阵 |
| 3 | [全链路指南](docs/fullstack-guide.md) | 一个算子从源码到推理的完整流程 |
| 4 | [测试体系](docs/testing.md) | 验证层级 · 黄金 · 漂移 · 一致性 |
| 5 | [性能回归追踪](docs/performance-regression.md) | 结构化记录 · 入库基线 · 双档门禁 |
| 6 | [AI 生成算子接入](docs/ai-intake.md) | KernelGen/KernelBench 产物 → 三级验证 |
| 7 | [已知问题](docs/known-issues.md) | 12 条实测记录 + 检测方法 |

深入主题: [A1](docs/route-a1-aten.md) / [A2](docs/route-a2-dispatch.md) / [B](docs/route-b-vendor.md) 路线详解 ·
[设备接入](docs/device-profiles.md) · [报告](docs/reporting.md) ·
[性能回归](docs/performance-regression.md) · [AI 生成算子](docs/ai-intake.md) ·
[样例索引](examples/README.md)（14 个可运行样例 + 定位图）

## 许可

[Apache-2.0](LICENSE) · [贡献指南](CONTRIBUTING.md) · [更新日志](CHANGELOG.md)
