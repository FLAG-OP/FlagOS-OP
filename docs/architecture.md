# 体系结构

## 设计总览: 三条路线 × 两级测试

| | kernel 层（硬件语言直测） | op 层（注册/分发/拦截） | framework 层（真实推理） |
|---|---|---|---|
| **A1** Triton→aten | 精度+哨兵+性能 | aten 注册+拦截 | 子进程注入+前缀恒等+黄金回归 |
| **A2** Triton→dispatch | 精度+哨兵+性能 | dispatch 注册+策略切换 | vendor 注入+前缀一致率 |
| **B** 厂商→vendor | **厂商 kernel 直测+C++ JIT+哨兵** | vendor 注册/选择 | 拦截计数+前缀恒等+黄金回归 |

另有 `--consistency`: 同算子 L0 直调 ↔ L2 dispatch 张量级一致矩阵。

## 为什么是三条路线

FlagOS 的算子替换发生在**两个层次**，加上芯片层共三条路线:

| 层次 | 算子类型 | 机制 |
|---|---|---|
| torch 层 | aten 算子（add/gelu/silu...） | `torch.library.Library("aten","IMPL").impl()` 按 dispatch key 注册 |
| vLLM 层 | 融合算子（silu_and_mul/rms_norm...） | FlagOS 自研 OpManager / OpRegistry / policy |
| 芯片层 | 厂商 C++/SDK kernel | vendor backend（Backend 子类 + OpImpl VENDOR 注册） |

FlagOS **没有自有 kernel 语言**——编程层复用 Triton（+厂商 kernel），
自研的是"分发"与"可移植"。

## 目录结构

```
flagos-op-templates/
├── run.py                  统一矩阵入口（--route/--level/--device/--all）
├── configs/devices/        设备 profile（芯片泛化的核心）
├── common/                 设备加载/探测 + 输入 scheme + 参考实现
├── routes/                 三条路线的正式实现
├── tests/                  op_level（三段式） + framework_level（真实推理）
├── examples/               9 个矩阵格的可运行样例（kernel/op 层自包含）
├── injection/              A1 框架级的 sitecustomize 跨进程注入桥
├── inputs/                 声明式输入模板（spec.yaml）
├── golden/                 跨设备黄金输出（多快照+共识前缀）
├── scripts/                一键脚本 + 黄金构建 + 漂移实验 + 输入生成
└── results/                每格结果 JSON（gitignore）
```

## 矩阵入口约定

- 所有测试模块统一签名 `run(profile) -> bool`
- 设备参数全部来自 profile——测试代码里没有硬编码设备串/卡号/厂商库名
- 框架级测试由 harness 拉起 vLLM 子进程（`_vllm_runner.py`），
  引擎参数、可见卡、注入环境变量按 profile 与路线自动组装
