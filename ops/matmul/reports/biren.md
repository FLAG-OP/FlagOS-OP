# matmul 壁仞 Biren166M 平台报告

状态：**PARTIAL / FP32、BF16 支持 dtype 完整黄金及 torch-level 三层通过，
非 Triton、FlagGems 或完整平台验收**

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
| Triton | 当前运行时未安装 |
| FlagGems | 当前运行时未安装 |

## 2. 设备与执行路径

- 设备类型是 `supa`，不是 `cuda` 或 `xpu`；
- dispatch key 为 `AutogradPrivateUse1`；
- 当前使用 A1 + torch-level 实现；
- `torch_br` 与 SDK 的 `br_generator` 共同注册 `supa` 设备；两者由
  `configs/devices/biren.yaml` 的运行时模块列表统一加载；
- 当前运行时没有 Triton 和 FlagGems，因此本平台未声称两者的能力通过。

因为通用 `mm/bmm` autograd 路径在部分形状上不可靠，Biren 后端使用
`torch.ops.supa.br_matmul_with_output` 作为计算原语。该接口已提示未来
迁移到 `br_fused_matmul_transpose`，属于后续维护项。

torch-level 直调使用显式一阶 autograd 包装，反向复用同一形状规则计算
`dA`、`dB` 并归约广播维度；A1 再通过 `AutogradPrivateUse1` 接管
`aten::matmul`。空张量和零 K 不调用厂商 GEMM，避免读取零长度指针或
未定义内存。

## 3. 支持 dtype 完整精度矩阵

测试入口：`script/check_accuracy.py`。

覆盖：

- 16 种 shape × FP32/BF16 × 3 seed = 96 组；
- 2-D 非整 tile/regular 矩阵、向量点积、矩阵 × 向量、向量 × 矩阵；
- 批量矩阵、批量广播、非连续输入；
- 零 K、空 M/N/batch、零输入、大数、tiny 和 `K=1`；
- 每组前向、左输入梯度和右输入梯度；
- 直调及 A1 两条入口。

| 项目 | 结果 |
|---|---|
| torch-level 直调黄金 | 96/96 PASS |
| A1 黄金 | 96/96 PASS |
| A1 命中 | 96/96 |
| 最坏前向/双梯度相对误差 | 6.67e-3 |

FP16：

```text
STATUS_NOT_SUPPORTED:
For br10x, matmul input0 only support bf16 or fp32 or s8 or u8 now!
```

当前 Biren166M/BR10x 栈不支持 FP16 matmul，错误会以 `SIGABRT` 终止
进程，因此 FP16 在独立子进程中探测，并未记为通过。故完整范围是
FP32/BF16 `96/96`，不是三 dtype `144/144`。

## 4. 三层与应用消费者

| 层级 | 入口 | 结果 |
|---|---|---|
| 低层直调 | `check_accuracy.py --impl torch --dtype float32 --dtype bfloat16` | 96/96 |
| A1 黄金 | `check_accuracy.py --impl a1-torch --dtype float32 --dtype bfloat16` | 96/96，命中 96 次 |
| A1 op | `test/op_level.py biren --backend torch --dtype float32 --dtype bfloat16` | 8 次命中；三种模式各 1 次；CPU 隔离通过 |
| framework | `test/biren_framework.py` | 两次 matmul 命中；CPU 对照输出和三个梯度最大误差 `9.45e-5` |
| smoke | `test/biren_smoke.py` | 7 shape × FP32/BF16 直调 14/14；三种模式各 14 次命中 |

`biren_framework.py` 不使用 FlagGems GELU，改用 PyTorch GELU，并以
确定性 CPU 消费者作为输出和梯度基线；壁仞原生批量反向存在广播错误，
因此没有把故障的原生路径当作 A1 的正确性基线。

## 5. 明确边界

- FP16：BR10x 明确不支持，独立子进程退出码为 `-6`；
- 原生 1-D 向量：`torch.matmul` 报 matrix operands 至少 2-D，算子的
  形状整理路径覆盖该语义；
- Triton：当前 `birensupa-pytorch` 运行时 `find_spec("triton")` 为
  `None`，没有 Triton 编译或 kernel 证据；
- FlagGems：当前运行时 `find_spec("flag_gems")` 为 `None`，没有
  FlagGems 接入或共存证据；
- 原生批量反向：部分 broadcast shape 触发 BIREN SUDNN 广播错误，不计作
  通过，也没有作为 A1 的真实性判据。

边界探针：`test/biren_boundaries.py`。

## 6. 初始性能

当前 profile 只登记 `torch` 性能入口。12 个非空 shape × FP32/BF16 ×
前后向共 48 条，每条独立进程、warmup 20、iters 100；48/48 成功、0 跳过，
计时前最大检查误差 `7.0e-3`。

| dtype | 方向 | 范围 ms | 中位数 ms |
|---|---|---:|---:|
| float32 | 前向 | 0.298–2.690 | 0.365 |
| float32 | 反向 | 0.715–5.471 | 0.800 |
| bfloat16 | 前向 | 0.283–2.420 | 0.350 |
| bfloat16 | 反向 | 0.683–5.006 | 0.772 |

BF16 `63×127 @ 127×65`：native `0.266696 ms`，direct
torch-level `0.380200 ms`，A1 `0.395256 ms`；A1 比直调多
`0.015056 ms`。同轮基线回读为 OK 48 / WARN 0 / FAIL 0，但它不是
第二次独立复跑。共享卡上的初始数据只用于后续回归，不作优化结论。

基线、48 条原始 JSON 和分发测量位于 `biren_evidence/`。

## 7. 复现

进入容器后：

```bash
source /usr/local/birensupa/br_container_tools/brsw_set_env.sh
cd /workspace/FlagOS-OP
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/test/biren_smoke.py
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/script/check_accuracy.py \
    --impl torch --device biren --dtype float32 --dtype bfloat16
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/script/check_accuracy.py \
    --impl a1-torch --device biren --dtype float32 --dtype bfloat16
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/test/op_level.py biren --backend torch \
    --dtype float32 --dtype bfloat16
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/test/biren_framework.py
PYTHONPATH=/workspace/FlagOS-OP:$PYTHONPATH \
  python3 ops/matmul/test/biren_boundaries.py
```

原始日志见 [biren_evidence](biren_evidence/)。

## 8. 尚未完成

1. Triton 安装、Biren backend 支持及 kernel 级验证；
2. FlagGems 接入和共存验证；
3. FP16 等待 BR11x 或后续 SDK 支持；
4. 原生批量反向稳定性及厂商接口迁移；
5. 48 条性能记录的独立复跑和 profiler 分解；
6. 真实引擎 C2。
