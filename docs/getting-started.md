# 快速开始

## 环境要求

- FlagOS 生态: FlagGems + vllm-plugin-FL + vLLM
  （本库基于 4.2.1rc0 / 0.1.0 / 0.13.0 实测）
- Triton（随 FlagGems 提供）
- 设备 profile: `configs/devices/` 下有你的芯片对应 YAML
  （参考实例 p800-kunlunxin；新芯片见 device-profiles.md 3 步接入）

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

## 结果在哪

- 每格结果 JSON: `results/<device>_<route>_<level>.json`
- 黄金输出: `golden/<device>_golden.json`
- 控制台有逐格 PASS/FAIL 摘要

## 换设备

```bash
DEVICE=nvidia ./scripts/run_all.sh            # 薄封装方式
python3 run.py --all --device <profile名>     # 直接方式
```

没有对应 profile 时会列出可用项；接入新芯片见 `device-profiles.md`。
