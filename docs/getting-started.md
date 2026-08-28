# 快速开始

[← 返回文档中心](index.md)

> 这一页帮你用 5 分钟跑通第一个验证。如果还不清楚这个库在做什么，
> 先读[体系结构](architecture.md)；想完整开发一个算子，跟着
> [全链路指南](fullstack-guide.md)走。

## 环境要求

- FlagOS 生态: FlagGems + vllm-plugin-FL + vLLM + Triton（版本随共享镜像提供）
- P800 共享镜像: 实测版本锁定在
  [configs/env/p800-kunlunxin.lock.yaml](../configs/env/p800-kunlunxin.lock.yaml)，
  用下面的命令核对当前容器是否仍与实测基线一致:

```bash
python3 scripts/check_env.py --device p800-kunlunxin --require-model
```

- Triton（随 FlagGems 提供）
- 设备 profile: `configs/devices/` 下有你的芯片对应 YAML
  （参考实例 p800-kunlunxin；新芯片见 device-profiles.md 4 步接入）

## 5 分钟流程

```bash
cd /workspace/flagos-op-templates
# 1. 看有哪些设备与矩阵格
python3 run.py --list
# 2. 跑最轻的一格（示例用参考 profile，换成你的芯片名即可）
python3 run.py --route a1 --level op --device p800-kunlunxin
# 3. 跑一个框架级样例（约 2 分钟，含真实 vLLM 推理）
python3 examples/a1-framework/example.py
# 4. 全矩阵（9 格 + consistency，时长视设备而定）
DEVICE=p800-kunlunxin ./scripts/run_all.sh
```

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
