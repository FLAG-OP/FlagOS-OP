# Ascend910 真实 vLLM C2 证据 — empty_strided

- **级别**: **C2**（真实 vLLM 前向、**生成阶段**命中，输出与基线逐位一致）
- **栈**: `vllm-plugin-fl` / `PlatformFL`（**不是** vLLM-Ascend）
- **方法**: vLLM **EngineCore（多进程）** 三段运行 `baseline` / `init-only` / `injected`；
  在 worker 内 patch `WorkerFL.execute_model`，对每次前向取 `CALL_COUNT` 前/后快照，
  区分 **init 命中** 与 **generate 阶段命中**；退出时一次性 dump（避免热路径 I/O）。
- **模型**: `scripts/c2_enginecore/make_model.py`（seed=0，hidden=64/heads=4/layers=2/head_dim=16）
- **环境**: driver 25.5.0 / CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0 /
  flagtree 3.5.1 / vllm 0.20.2+empty / vllm_fl g8b059122e / SOC Ascend910_9382 /
  `ASCEND_RT_VISIBLE_DEVICES=1`
- **结果**: baseline/injected token 均 `[7, 21, 1, 18, 10, 23, 21, 1]`（逐位一致）；
  **init=98 · generation=11 · final=109**；`passed=true`
- **证据**: [c2_ascend910/summary.json](c2_ascend910/summary.json) + `baseline/stdout` 日志 +
  每 PID `count_<pid>.json`（init/generation/final）
- **复现**: 见 [scripts/c2_enginecore/README.md](../../../scripts/c2_enginecore/README.md)
