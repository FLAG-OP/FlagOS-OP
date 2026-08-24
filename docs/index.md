# FlagOS 算子开发模板库 · 文档中心

<a id="top"></a>
## 目录

1. [快速开始](getting-started.md) — 5 分钟跑通第一格
2. [体系结构](architecture.md) — 层次图 · 3×3 矩阵 · 目录导航
3. 路线详解
   - [路线 A1: Triton → aten](route-a1-aten.md)
   - [路线 A2: Triton → dispatch](route-a2-dispatch.md)
   - [路线 B: 厂商语言 → vendor](route-b-vendor.md)
4. [测试体系](testing.md) — kernel/op/framework 三层 · 黄金 · 漂移 · 一致性
5. [全链路开发指南](fullstack-guide.md) ⭐ — 一个算子从源码到真实推理
6. [设备接入](device-profiles.md) — 芯片泛化 · 新芯片 3 步接入
7. [报告与环境快照](reporting.md) — 算子开发报告模板
8. [已知问题](known-issues.md) — 参考 case 记录 + 通用检测方法

## 体系一图

```
                 FlagOS 算子开发模板库
                        │
      ┌─────────────────┼──────────────────┐
      │                 │                  │
 三条实现路线        三层验证            支撑设施
      │                 │                  │
 ┌────┴───┐      ┌─────┼──────┐     ┌─────┴─────┐
 A1       A2      kernel op  framework  设备profile  黄金输出/漂移实验
 aten  dispatch    直测  注册  真实推理  (芯片泛化)   报告模板/环境快照
 路线   路线  B                 │
              vendor            └─ 全链路: [b-fullstack](../examples/b-fullstack/)
              路线                 (同一算子贯穿三层)
```

## 按任务找入口

| 我想… | 去哪 |
|---|---|
| 跑通第一个测试 | [快速开始](getting-started.md) |
| 理解三条路线区别 | [体系结构](architecture.md#routes) |
| 开发 torch 算子替换 | [路线 A1](route-a1-aten.md) |
| 开发 vLLM 融合算子 | [路线 A2](route-a2-dispatch.md) |
| 接入厂商 C++ kernel | [路线 B](route-b-vendor.md) |
| 完整走一遍开发到上线 | [全链路指南](fullstack-guide.md) |
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
| vendor backend | 厂商 kernel 的 Python 接入层 | [路线 B](route-b-vendor.md) |
