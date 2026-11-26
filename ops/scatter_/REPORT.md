# scatter_ 交付报告

2026-11-26，Dashener2：Triton 级 / A1，p800-kunlunxin-gems5。
覆盖无归约 src/value 重载；推理用途。src 补齐真实 vLLM C2；value 为 C1。

| 检查 | 实测结果 |
|---|---|
| kernel 直测 | 41/41 |
| 额外边界 | 11/11，含越界拒绝、重复索引、int64 大标量、4D/5D、大块 |
| A1 | 41/41，返回对象及版本计数检查通过 |
| mini consumer | token-selection 输出精确一致，命中 1 次 |
| 真实 vLLM C2 | 随机权重 Qwen3，src 生成阶段命中 8 次，两请求共 16 token 精确一致 |
| 黄金 | CPU 生成 + P800 Triton 回放 41/41，SHA256 校验 |
| 稳定性 | 修复尾块后 kernel/A1 连续 10 轮通过 |
| 环境锁 | check_env p800-kunlunxin-gems5 = OK |
| CPU | 元数据 pytest 7 项、Triton 解释器 34 项、包化 17/17 |
| 全库基础设施 | tests/unit 186/186；evidence_check 通过 |
| 性能 | 7×1031 为原生速度的 0.062x；128×4096 为 0.051x |

结构化结果：`reports/test-p800.json`、`reports/perf-p800.json`。
真实框架结果：`reports/vllm-c2.json`，复现及范围见 `reports/vllm.md`。
当前性能不能满足加速用途；本次作为正确性/注册/兼容性实现提交审阅。

## 复现环境

专用容器 `shener-scatter-p800`，仅映射物理 `/dev/xpu0` 和控制节点。
逻辑 cuda:0；CPU 4 核 / 内存 16GiB；其他卡不映射、不操作其他容器。
镜像：`harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4`。
本机镜像 ID `e180a0dd4267`（不是 registry digest）。
Python 3.10.18、torch 2.9.0+cu129、XMLIR 1.0.0.1、
Triton 3.0.0+f69a5406、FlagGems 5.0.0。
宿主 P800 Driver 5.0.21.47 / XPU-RT 5.0.21。

命令见 README。容器 Python 路径为
`/root/miniconda/envs/python310_torch29_cuda/bin/python`。
profile 的默认 cuda:1 被命令行覆盖为 cuda:0，未修改共享 profile。

CPU 验证独立环境：Python 3.12、torch 2.15.0+cpu、Triton 3.9.0。
CPU 解释器跳过 bf16 和空索引；P800 41 用例包含它们。
