# 寒武纪 matmul 证据溯源

## 1. 边界

本目录中的功能、精度和性能结果是 2026-11-10 至 2026-11-11 在共享容器
`mlu-baseline` 中采集的原始记录。采集时工作树记录为：

```text
base commit: 4df994d31962c3135a6dcb22a9fbc313761e90ac
git_dirty: true
device: mlu590
torch_device: mlu:0
```

由于当时没有保存完整 dirty patch，无法从原始 JSON 单独证明事后提交与
采集工作树逐字节相同。本文件给出提交快照的源码身份，作为关联证据；
不能把该映射写成对历史脏工作树的独立认证。

## 2. 关联源码快照

采集后的寒武纪改动整理在 PR #53。与设备结果直接相关的文件在提交
`330732400700603af78f159b7aba4c7436cbf2e6` 中的 Git blob 身份如下：

| 文件 | Git blob SHA-1 |
|---|---|
| `ops/matmul/register.py` | `5eb5ce62656c1d1b96f67c6989332f437a8cfc27` |
| `ops/matmul/kernel/triton_level.py` | `4d98ebfdec0f43c01d44dcc4106f8823fb25de3d` |
| `ops/matmul/kernel/_shape.py` | `d9fd72c1d5fc3bb212a5ba49e15c0d51a72ea446` |
| `ops/matmul/reference.py` | `1829343a740f7adbe69d0d5209da3214cd4cefda` |

复核命令：

```bash
git rev-parse 3307324:ops/matmul/register.py
git rev-parse 3307324:ops/matmul/kernel/triton_level.py
git rev-parse 3307324:ops/matmul/kernel/_shape.py
git rev-parse 3307324:ops/matmul/reference.py
```

## 3. 采集环境

| 项 | 已确认值 |
|---|---|
| 服务器 / 容器 | `tza-0a06-ai01-em9` / `mlu-baseline` |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cpu |
| torch_mlu | 1.32.2+torch2.10.0 |
| Triton | 3.4.0+mlu2.0.3 |
| FlagGems | 5.3.5 源码 |
| FlagGems 源码路径 | `/root/FlagGems-src/src` |
| 逻辑设备 | `mlu:0` |
| dispatch key | `AutogradPrivateUse1` |

## 4. 尚未恢复的身份信息

以下字段在原始记录中缺失，保持 PENDING，不能补猜：

| 字段 | 状态 | 影响 |
|---|---|---|
| 物理 MLU 卡号 / 拓扑映射 | PENDING | 无法证明独占和固定同一物理卡 |
| 厂商驱动与 Neuware 版本 | PENDING | 无法完成厂商栈锁定 |
| FlagGems 源码 commit | PENDING | 只记录到 5.3.5 源码目录 |
| 采集时 dirty patch 的完整哈希 | PENDING | 只能使用提交快照作关联证据 |
| 锁定环境下的独立性能复跑 | PENDING | 当前数字仅作兼容性基线 |
| 真实推理引擎 C2 | PENDING | mini consumer 不能代替真实 vLLM |

## 5. 下一轮采集要求

在锁定环境复验时必须同时保存：

1. `git status --porcelain` 和 `git diff --binary`；
2. 所有相关源码文件的 SHA256；
3. FlagGems `git rev-parse HEAD`；
4. `cnmon` 或厂商等价工具的卡号、拓扑和利用率；
5. 厂商驱动、Neuware、torch_mlu 和 Triton 的实际版本；
6. 同一进程的 `torch_mlu`、`torch.mlu.device_count()` 和设备名；
7. grad、no_grad、inference_mode 三次实际命中计数；
8. 新采样记录与归档 baseline 的 `perf_compare` 输出。
