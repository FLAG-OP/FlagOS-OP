# FlagOS-OP

FlagOS-OP 是一套自定义算子的开发模板库。写一个算子并不难，难的是
把它安全地用起来：实现是否正确、注册之后是否真的被选中、放进真实
推理会不会破坏输出、性能有没有退化——这个库把这些验证步骤做成
可复用的模板，照着样例替换自己的 kernel 即可。

整个库围绕一个 3×3 矩阵组织：A1 / A2 / B 三条路线决定算子从哪里
接入，算子库层 / 框架层 / 应用层三个位置决定在哪里验证。芯片差异
全部收在设备 profile 里，测试代码不感知具体硬件。

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
cp -r templates/operator ops/my_op/         # ← 新算子从这里开始
python3 scripts/perf_compare.py --device p800-kunlunxin   # 性能回归门禁
python3 scripts/intake_validate.py --validate-only        # AI 生成算子契约
```

## 文档

完整文档在 [docs/](docs/index.md)，建议阅读顺序:

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [快速开始](docs/getting-started.md) | 5 分钟跑通第一格 |
| 2 | [体系结构](docs/architecture.md) | 物理栈 · 开发级别 · 路线 · 3×3 矩阵 |
| 3 | [全链路指南](docs/fullstack-guide.md) | 七步流水线 · 每步抽象动作 × 样例映射 |
| 4 | [测试体系](docs/testing.md) | 验证层级 · 黄金 · 漂移 · 一致性 |
| 5 | [性能回归追踪](docs/performance-regression.md) | 结构化记录 · 入库基线 · 双档门禁 |
| 6 | [验收标准](docs/acceptance.md) | 精度/性能口径 · Must/Should 分级 · 重验 runbook |
| 7 | [算子集成指南](docs/integration.md) | 交付路径选择 · FlagGems 贡献 · 分发开销实测 |
| 8 | [AI 生成算子接入](docs/ai-intake.md) | KernelGen/KernelBench 产物 → 三级验证 |
| 9 | [已知问题](docs/known-issues.md) | 18 条实测记录 + 检测方法 |

深入主题: [A1](docs/route-a1-aten.md) / [A2](docs/route-a2-dispatch.md) / [B](docs/route-b-vendor.md) 路线详解 ·
[设备接入](docs/device-profiles.md) · [报告](docs/reporting.md) ·
[性能回归](docs/performance-regression.md) · [AI 生成算子](docs/ai-intake.md) ·
[样例索引](examples/README.md)（14 个可运行样例 + 定位图）

## 仓库分区

| 分区 | 内容 | 何时进入 |
|---|---|---|
| [templates/](templates/) | **算子样板** + 报告模板 | 开发新算子的起点 |
| [ops/](ops/) | **已开发算子**（样板同构，每个算子一个目录） | 开发完成、验证全绿后进入 |
| [routes/](routes/) + [tests/](tests/) | 三路线正式实现 + 3×3 矩阵测试 | 实现稳定后合入 |
| [examples/](examples/) | 14 个教学样例（按路线×层级定位） | 学习与对照 |
| [intake/](intake/) + [perf/](perf/) | AI 生成算子通道 + 性能基线 | 自动化环节 |
| [docs/](docs/) + [reports/examples/](reports/examples/) | 文档 + 示例报告 | 全程参考 |

## 许可

[Apache-2.0](LICENSE) · [贡献指南](CONTRIBUTING.md) · [更新日志](CHANGELOG.md)
