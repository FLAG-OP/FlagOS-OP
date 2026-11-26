# 真实 vLLM 验证（Ascend910，2026-11-26）

结论：**PASS，scatter_.src 的 C2**。真实 `vllm.LLM.generate` 在
top-k/top-p 采样路径命中仓库实现 **8 次**，两个请求各 8 个输出 token 与
原生基线精确一致。value 重载未在本次 vLLM 路径命中，仍为 C1。

## 范围

- 固定随机种子 20261126 创建的 Qwen3 随机权重：2 层、hidden 128、
  intermediate 256、2Q/1KV、head_dim 64、vocab 1024、bf16。
  真实模型加载、prefill、decode、采样；直接传 token IDs，跳过 tokenizer。
  不是预训练模型质量验证，不代表大模型、多卡或服务吞吐验证。
- 栈：vLLM 0.20.2 + `vllm-plugin-fl`（PlatformFL，device npu）；
  torch/torch_npu，triton-ascend；`TASK_QUEUE_ENABLE=0`。
- TP=1、enforce_eager=True、V1 单进程（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）、
  `USE_FLAGGEMS=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`。基线/注入两组配置相同。
- temperature=0.8、top_k=32、top_p=0.9、采样 seed=1234、ignore_eos=True。
- 注入在引擎初始化之后、generate 之前进行（`AutogradPrivateUse1`，platform=ascend）；
  8 次命中不含启动/profile；首次命中完整调用栈落在 `apply_top_k_top_p_pytorch`
  → `logits.scatter_(dim=-1, index, src)`。

## kernelGen 说明

- 用户提供的 kernelGen 初稿为 TLE/DSA 版本（`tle.dsa.copy` 串行写回）；
  在本栈 `TritonToLinalgIncubated` 编译失败（41 例 40 失败）。
- 故改写为**纯 Triton**（按坐标计算 dst/src 偏移），语义不变；期间
  kernelGen MCP 后端不可用（“无法连接到后端服务”/504），未能在线迭代。

## 容器与设备

- 容器 `flagos-op-ascend-daizijian`，镜像
  `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`；
  `--runtime ascend --network host`，映射 `/dev/davinci0`、`/dev/davinci1` 等；
  `ASCEND_RT_VISIBLE_DEVICES=0,1`；仓库挂载到 `/workspace/flagos_op/FlagOS-OP`。
- `PYTHONPATH=/workspace/flagrt_stub:/workspace/flagos_op/FlagOS-OP`（shmem 桩）。

## 复现（容器内，仓库根目录）

```bash
python3 ops/scatter_/test/vllm_runner.py --mode prepare  --model /workspace/scratch/scatter-qwen3-tiny
python3 ops/scatter_/test/vllm_runner.py --mode baseline --model /workspace/scratch/scatter-qwen3-tiny --out ops/scatter_/reports/vllm-baseline.json
python3 ops/scatter_/test/vllm_runner.py --mode injected --model /workspace/scratch/scatter-qwen3-tiny --out ops/scatter_/reports/vllm-injected.json
python3 ops/scatter_/script/check_vllm.py ops/scatter_/reports/vllm-baseline.json ops/scatter_/reports/vllm-injected.json --out ops/scatter_/reports/vllm-c2.json
```

模型 config/权重 SHA256、输出 token 与首次命中调用栈均存入上述 JSON；
比较脚本校验模型指纹、采样参数、引擎配置一致，输出相等且命中 > 0。
