# 真实 vLLM 验证（2026-11-26）

结论：**PASS，scatter_.src 的 C2**。真实 `vllm.LLM.generate` 在
top-k/top-p 采样路径命中仓库实现 8 次，两个请求各 8 个输出 token 与
原生基线精确一致。value 重载未在本次 vLLM 路径命中，仍为 C1。

## 范围

- 固定随机种子 20261126 创建的 Qwen3 随机权重：2 层、hidden 128、
  intermediate 256、2Q/1KV、head_dim 64、vocab 1024、bf16。
- 真实模型加载、prefill、decode、采样；直接传 token IDs，跳过 tokenizer。
  不是预训练模型质量验证，不代表大模型、多卡或服务吞吐验证。
- vLLM 0.20.2，plugin-FL commit `38e7dbc20197e2db742c4e4c9687d36ea4df9900`；
  插件工作树无 tracked 改动，镜像内存在备份文件（untracked）。
- TP=1、enforce_eager=True、V1 multiprocessing=False、USE_FLAGGEMS=0。
  原生平台插件自带的 PyTorch top-k/top-p 路径，两组保持相同配置；
  没有为了制造命中替换采样消费者。
- temperature=0.8、top_k=32、top_p=0.9、采样 seed=1234、ignore_eos=True。
- 注入在引擎初始化之后、generate 之前进行；8 次命中不含启动/profile。
  单进程 v1 使注册和模型 worker 在同一进程，首次命中完整调用栈落盘。

## 容器与设备

容器 `shener-scatter-vllm`；镜像与 REPORT 中的 tele4 相同。
只映射 /dev/xpu0 和 /dev/xpuctrl；CUDA_VISIBLE_DEVICES=XPU_VISIBLE_DEVICES=0。
CPU 4 核、内存 16GiB、shm 2GiB，仓库挂载到 `/work/FlagOS-OP`。
**必须保留镜像的 `/workspace/vllm-plugin-FL`**：editable 安装依赖此路径；
把仓库覆盖挂载到 /workspace 会导致插件导入失败。

## 复现（容器内）

使用 `/root/miniconda/envs/python310_torch29_cuda/bin/python`，仓库根目录：

```bash
python ops/scatter_/test/vllm_runner.py --mode prepare --model /models/scatter-qwen3-tiny
python ops/scatter_/test/vllm_runner.py --mode baseline --model /models/scatter-qwen3-tiny --out ops/scatter_/reports/vllm-baseline.json
python ops/scatter_/test/vllm_runner.py --mode injected --model /models/scatter-qwen3-tiny --out ops/scatter_/reports/vllm-injected.json
python ops/scatter_/script/check_vllm.py ops/scatter_/reports/vllm-baseline.json ops/scatter_/reports/vllm-injected.json --out ops/scatter_/reports/vllm-c2.json
```

模型 config/权重 SHA256、输出 token 和首次命中调用栈均存入上述 JSON。
比较脚本验证模型指纹、参数、引擎配置一致，输出相等且实际命中大于零。
引擎和 distributed group 显式 shutdown；修复退出清理后重新运行基线及
注入，两者均正常退出并重新通过比较。
