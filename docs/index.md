# FlagOS 算子开发模板库 · 文档中心

<a id="top"></a>
## 目录

1. [快速开始](getting-started.md) — 5 分钟跑通第一格
2. [体系结构](architecture.md) — 层次图 · 3×3 矩阵 · 目录导航
3. 路线详解
   - [路线 A1: torch 算子替换](route-a1-aten.md)
   - [路线 A2: FlagOS 融合算子](route-a2-dispatch.md)
   - [路线 B: 厂商算子接入](route-b-vendor.md)
4. [测试体系](testing.md) — kernel/op/framework 三层 · 黄金 · 漂移 · 一致性
5. [全链路开发指南](fullstack-guide.md) ⭐ — 一个算子从源码到真实推理
6. [设备接入](device-profiles.md) — 新芯片接入
7. [报告与环境快照](reporting.md) — 算子开发报告模板
8. [已知问题](known-issues.md) — 参考 case 记录 + 通用检测方法

## 体系一图

```
  应用层    vLLM · transformers           ← L4 framework 验证
    ↓
  框架层    PyTorch + FlagOS 融合算子     ← L2 op 验证 · A1/A2 路线
    ↓
  编译层    Triton → 芯片编译栈           ← TR 开发层级
    ↓
  算子库层  FlagGems · 厂商 kernel        ← L0 kernel 验证 · B 路线 · FW/HW
    ↓
  硬件层    XPU · GPU · NPU              ← 设备 profile
```

完整物理栈图与概念映射见[根 README](../README.md#map)。## 按任务找入口

| 我想… | 去哪 |
|---|---|
| 跑通第一个测试 | [快速开始](getting-started.md) |
| 理解三条路线区别 | [体系结构](architecture.md#routes) |
| 开发 torch 算子替换 | [路线 A1](route-a1-aten.md) |
| 开发 vLLM 融合算子 | [路线 A2](route-a2-dispatch.md) |
| 接入厂商 C++ kernel | [路线 B](route-b-vendor.md) |
| 完整走一遍开发到上线 | [全链路指南](fullstack-guide.md)（4 个范本） |
| 理解每层验证断什么 | [测试体系](testing.md) |
| 接入新芯片 | [设备接入](device-profiles.md#onboard) |
| 写开发报告 | [报告指南](reporting.md) |
| 排查数值问题 | [已知问题](known-issues.md#method) |
| 看可运行样例 | [examples/](../examples/README.md) |

## 术语速查（带链接）

| 术语 | 含义 | 详见 |
|---|---|---|
| 设备 profile | 芯片差异的 YAML 声明 | [设备接入](device-profiles.md) |
| [kernel 层](testing.md#kernel-level) | 硬件语言 kernel 直测 | 测试体系 |
| [哨兵检查](testing.md#sentinel) | 检测 kernel "不写输出"类 bug | 测试体系 |
| [黄金输出](testing.md#golden) | 多快照共识回归锚点 | 测试体系 |
| [跨层一致性](testing.md#consistency) | 同算子 L0↔L2 张量级比对 | 测试体系 |
| PER_OP | 按算子钉选后端的策略 | [路线 A2](route-a2-dispatch.md) |
| 厂商算子接入 | 厂商 kernel 的 Python 接入层 | [路线 B](route-b-vendor.md) |
| 开发层级 | kernel 写在哪一层: 框架层 FW / Triton 层 TR / 硬件语言层 HW | [体系结构](architecture.md#levels) |
