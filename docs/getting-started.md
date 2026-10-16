# 快速开始

[← 返回文档中心](index.md)

> 这一页帮你从最小契约验证开始跑通第一个验证。如果还不清楚这个库在做什么，
> 先读[体系结构](architecture.md)；想完整开发一个算子，跟着
> [全链路指南](fullstack-guide.md)走。
> 仓库下载、`PYTHONPATH`、FlagGems 源码 / 运行时配合方式见
> [安装与 FlagGems 配合使用](installation.md)。

## 环境要求

| 需要什么 | 说明 |
|---|---|
| 契约/算子层验证 | Python、PyTorch 和仓库依赖；不需要模型或 vLLM。CPU profile 可作为最小参考路径 |
| 目标设备验证 | 对应设备 runtime、Triton/厂商依赖和 `configs/devices/` profile；版本以 [软硬件环境版本汇总](environments.md) 为准 |
| 真实引擎验证 | 在目标 profile 上额外需要 FlagGems、vllm-plugin-FL、vLLM、模型和 Triton；先运行 `check_env.py --require-model` |

profile 是运行配置声明，不等于该平台上的全部算子都已有交付证据；支持范围以
[算子 / 平台矩阵](operator-catalog.md) 和对应算子报告为准。

## 5 分钟流程

```bash
# 以下命令均在 FlagOS-OP 仓库根目录执行
# 1. 看有哪些设备与矩阵格
python3 run.py --list
# 2. 最小契约/注册验证（CPU，不需要模型）
python3 run.py --route a1 --level op --device cpu
# 3. 目标设备上的算子层验证（将 profile 替换为已存在的名称）
python3 run.py --route a1 --level op --device <profile>
# 4. 真实引擎验证前先核对软件栈和模型
python3 scripts/check_env.py --device <profile> --require-model
# 5. 目标 profile 的完整矩阵（仅在环境和模型就绪后运行）
DEVICE=<profile> ./scripts/run_all.sh
```

三条路径的边界如下：

- **最小路径**：用于确认仓库可导入、参考实现和注册契约可工作；结果不能代表目标硬件性能。
- **目标设备路径**：用于确认指定 profile 的 kernel、注册和分发；只报告该 profile 已运行的层级。
- **真实引擎路径**：使用 `examples/a1-framework/example.py <profile>` 或
  `run.py --level framework`，报告必须注明真实 vLLM 还是 mini consumer，以及引擎版本。

第 2 步成功时你会看到类似输出（关键是最末的 `PASS`）:

```
  精度: 9/9 组合 PASS
  拦截: torch.relu -> Triton kernel
  CELL RESULT: PASS (5.8s)
```

如果 FAIL，先看控制台最后一段 traceback——每格测试都会把失败
原因打印出来，且结果 JSON（`results/`）里留有 `error` 字段可回查。

## 结果在哪

- 每格结果 JSON: `results/<device>_<route>_<level>.json`
- 黄金输出: `golden/<device>_golden.json`
- 性能运行记录: `results/perf/runs/<device>/`（对比基线:
  [性能回归追踪](performance-regression.md)）
- AI 生成算子报告: `results/intake/`（见 [AI 生成算子接入](ai-intake.md)）
- 控制台有逐格 PASS/FAIL 摘要

## 换设备

```bash
DEVICE=nvidia ./scripts/run_all.sh            # 薄封装方式
python3 run.py --all --device <profile名>     # 直接方式
```

没有对应 profile 时会列出可用项；接入新芯片见 `device-profiles.md`。

---

**下一步**: [体系结构](architecture.md)——理解物理栈、路线与开发级别的设计。
