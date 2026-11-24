# matmul 壁仞 Biren166M 平台报告

状态：**MINIMAL / torch-level A1 最小闭环通过，非完整平台验收**

壁仞移植与复验：赵俊堯、张哲、张子煜、赵炳凯。

## 1. 环境

| 项 | 实际值 |
|---|---|
| 日期 | 2026-11-24 |
| 服务器 | `wz-server008` |
| 容器 | `tyanfeng-matmul-supa` |
| 镜像 | `birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx` |
| 设备 | Biren166M，逻辑设备 `supa:0` |
| 可见卡 | 1 张，Bus `00000000:03:00.0` |
| 显存 | 65024 MB |
| Python | 3.10 |
| PyTorch | 2.8.0+cpu |
| torch_br | 1.11.0.20800+br1xx |
| SUPA / Driver | 1.12 / 1.12.0 |
| Triton | 3.6.0，但无 Biren active driver |
| FlagGems | 5.3.5 源码无法自动识别 Biren 设备 |

## 2. 设备与执行路径

- 设备类型是 `supa`，不是 `cuda` 或 `xpu`；
- dispatch key 为 `AutogradPrivateUse1`；
- 当前使用 A1 + torch-level 实现；
- Triton 3.6 没有可用的 Biren driver，因此本平台未声称 Triton kernel 通过；
- 显式导入 SDK 的 `br_generator` 后，原生与厂商矩阵乘运行正常。

因为通用 `mm/bmm` autograd 路径在部分形状上不可靠，Biren 后端使用
`torch.ops.supa.br_matmul_with_output` 作为计算原语。该接口已提示未来
迁移到 `br_fused_matmul_transpose`，属于后续维护项。

## 3. 最小功能矩阵

测试文件：`test/biren_smoke.py`。

覆盖：

- 2-D 非整 tile 矩阵；
- 向量点积；
- 矩阵 × 向量；
- 向量 × 矩阵；
- 批量矩阵；
- 批量广播；
- `K=1`；
- FP32、BF16；
- 前向、左右输入梯度；
- grad、no_grad、inference_mode；
- CPU 隔离。

| 项目 | 结果 |
|---|---|
| 原生矩阵乘（2 种 dtype） | 2/2 PASS |
| 直调 torch-level（7 shape × 2 dtype） | 14/14 PASS |
| A1 grad 命中 | 14/14 |
| A1 no_grad 命中 | 14/14 |
| A1 inference_mode 命中 | 14/14 |
| CPU 隔离 | PASS |
| 最坏相对误差 | 7.519e-3 |

FP16：

```text
For br10x, matmul input0 only support bf16 or fp32 or s8 or u8 now
```

当前 Biren166M/BR10x 栈不支持 FP16 matmul，因此没有把 FP16 记为通过。

## 4. 尚未完成

1. Triton 编译与 kernel 级测试；
2. FP16 硬件/SDK 支持确认；
3. FlagGems 接入与 framework 三层；
4. 完整 144 组黄金；
5. 性能基线；
6. 真实引擎 C2。

## 5. 复现

进入容器后：

```bash
source /usr/local/birensupa/br_container_tools/brsw_set_env.sh
cd /workspace/FlagOS-OP
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/test/biren_smoke.py
```

原始日志见 [biren_evidence](biren_evidence/)。
