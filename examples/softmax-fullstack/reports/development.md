# 算子开发报告: softmax

| 项 | 值 |
|---|---|
| 算子名称 | `softmax`（最后一维） |
| 实现路线 | **A1** torch 算子替换（aten `_softmax`） |
| 开发级别 | **Triton 级**（自研设备码） |
| 目标设备 | `p800-kunlunxin`（XPU，经 XMLIR 呈现为 cuda:1） |
| 开发者 | FlagOS-OP（教学样例） |
| 日期 | 2026-08-28 |
| 报告状态 | **定稿** |

---

## 1. 环境配置

### 1.1 硬件

设备 profile: **p800-kunlunxin**（vendor=kunlunxin，
torch_device=cuda:1，visible=1,2，dispatch_key=CUDA，TP=2）。

### 1.2 软件栈（`scripts/check_env.py` 核对通过，7/7）

| 组件 | 版本 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.0+cu129 |
| Triton | 3.0.0+03c4c9be |
| FlagGems | 4.2.1rc0 |
| vLLM | 0.13.0 |
| vllm-plugin-FL | 0.1.0 |
| transformers | 4.57.1 |

完整锁定见 [configs/env/p800-kunlunxin.lock.yaml](../../../configs/env/p800-kunlunxin.lock.yaml)。

## 2. 算子定义

- **语义**: 最后一维 softmax；参考实现 [reference.py](../reference.py)
  （fp32 计算后 cast 回输入 dtype，与仓库级 `accuracy_report.py` 同口径）
- **接口**: `softmax(x: Tensor, dim=-1) -> Tensor`，仅支持最后一维
- **数值规格**: fp32 容差 1e-5（abs），bf16/fp16 容差 1e-2（abs）；
  参考与实现的中间计算均为 fp32
- **shape 清单**: 必含非 BLOCK 整倍数维度（3072/5000/5120）——
  [#15a](../../../docs/known-issues.md) 漏网教训的固化

## 3. 实现说明

### 3.1 路线选择

宿主是 `aten::_softmax`（标准 torch 算子），按[集成指南](../../../docs/integration.md)
的判断走 **A1**：`torch.library.Library("aten","IMPL").impl("_softmax", ...)`
按 dispatch key CUDA 注册，任何 `F.softmax` 调用自动拦截。

### 3.2 kernel 设计与两条硬约束

三遍流式归约（Pass1 行最大 → Pass2 指数和 → Pass3 归一化写回），
支持 N > BLOCK。两条本栈实测硬约束决定了最终形态:

1. **尾块 pad（#15a）**: N 非 BLOCK 整数倍时，XPU 后端"masked load +
   `tl.sum`"产生污染，`other`/`tl.where` 三种防护实测全部无效——
   wrapper 将输入 pad 到 2048 整数倍（实体填充 -60000，exp 后贡献 0），
   kernel 全程无 mask；整数倍 N 零开销
2. **固定 BLOCK_N=2048（#15b）**: 移除 `@triton.autotune`——本栈
   autotuner 会选出配置表外的非法 `num_warps=5`，同一 kernel
   不可复现地时对时错

### 3.3 三级实现状态

| 级别 | 状态 | 说明 |
|---|---|---|
| torch 级 | ✅ | [kernel/torch_level.py](../kernel/torch_level.py)（ATen 直通，对照实现） |
| Triton 级 | ✅ | [kernel/triton_level.py](../kernel/triton_level.py)（本报告主实现） |
| 硬件级 | ⬜ 置空 | 厂商库无同语义原语、XTC 未提供（[说明](../kernel/hardware_level/README.md)） |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ PASS | 24 组精度（8 shape × 3 dtype，含尾块回归）全过；哨兵确定性+输入敏感通过；最差 max_err=1.5e-5（bf16 组，容差 1e-2 内） |
| 框架层 | ✅ PASS | aten 拦截命中（calls=1），拦截路径 max_err=1.2e-4 |
| 应用层 | ✅ PASS | attention scores 消费者触发 3 次，输出 max_err=0 |
| 黄金回归 | ✅ PASS | 39 组黄金（gen_golden 按规格生成，sha256 索引），triton 实现 39/39 |
| 跨层一致性 | ◐ | 仓库级 `--consistency` 锚定 gelu_and_mul；本算子以应用层双跑等效覆盖 |

## 5. 性能

1024² bf16，短采样 20+100，子进程隔离（详见[性能分册](performance.md)）:

| 实现 | 延迟 | 相对 | 精度代价 |
|---|---|---|---|
| 自研 Triton | 0.060ms | 1.00x | 无 |
| FlagGems | 0.028ms | 2.1x 快 | **fp32 误差 1.5e-2（超差）** |
| 原生 ATen | **0.010ms** | 6.0x 快 | 无 |

三条用例已入[性能基线](../../../docs/performance-regression.md)回归门禁。
修复 #15 的正确性代价: 0.035→0.060ms（pad + 固定配置），换来全
dtype 精度最优。

## 6. 已知问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | [#13](../../../docs/known-issues.md) FlagGems gelu(tanh) 链接失败 | 与本算子无直接冲突，但说明本栈 FlagGems 局限 | 已记录 |
| 2 | [#15a](../../../docs/known-issues.md) 尾块归约污染 | 已修复（pad 方案）；换动态 shape 场景需保留防护 | 已修复 |
| 3 | [#15b](../../../docs/known-issues.md) autotuner 非法 num_warps | 已移除 autotune；栈升级后可复评 | 已规避 |

## 7. 结论与后续

**可交付**。三层验证全绿、精度三方最优、性能入回归门禁。后续方向:
单遍在线归约（running max/sum，Flash 风格）——在不降精度前提下
逼近 FlagGems 的 0.028ms。

## 附录: 复现命令

```bash
python3 examples/softmax-fullstack/example.py p800-kunlunxin   # 三层全绿
cd examples/softmax-fullstack
python3 script/gen_golden.py --device cpu                      # 39 组黄金
python3 script/check_accuracy.py --impl reference --device cpu # 39/39
python3 script/check_accuracy.py --impl triton --device p800-kunlunxin
cd ../../..
python3 scripts/accuracy_report.py --device p800-kunlunxin     # 三方精度
python3 scripts/perf_run.py --pattern softmax --device p800-kunlunxin
```
