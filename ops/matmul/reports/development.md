# 算子开发报告: matmul

| 项 | 值 |
|---|---|
| 算子名称 | `aten::matmul` |
| 实现路线 | **A1**：PyTorch/ATen 算子替换 |
| 开发级别 | **Triton 级**（自研 dot）；硬件级置空 |
| 目标设备 | Kunlunxin P800，物理 XPU 2 |
| 开发者 | 陶焱烽 |
| 日期 / 状态 | 2026-10-23 / PARTIAL：昆仑芯功能版本已完成，性能门禁尚未闭合 |

## 1. 环境配置

| 项 | 值 |
|---|---|
| 设备 | Kunlunxin P800，物理 XPU 2；`CUDA_VISIBLE_DEVICES=1,2`，逻辑 `cuda:1` |
| 容器 | 个人容器 `tyanfeng-flagos-p800-lock` |
| Python / Torch | 3.10.18 / 2.9.0+cu129 |
| Triton / FlagGems | 3.0.0+03c4c9be / 4.2.1rc0 |
| 配置 | `configs/devices/p800-kunlunxin.yaml`、`configs/env/p800-kunlunxin.lock.yaml` |

2026-10-14 环境锁检查为 `OK`。项目 profile 的正式映射下逻辑 `cuda:1`
指向物理 XPU 2；该卡空闲，黄金、三级和性能已在正式目标卡实测。
个人容器未安装 `pytest`；2026-10-23 的仓库 CPU 单测在本机临时环境执行，
结果见 [PR 自查记录](pr-readiness.md)。

## 2. 算子定义

### 2.1 语义与接口

```python
torch.matmul(a, b) -> torch.Tensor
```

支持 FP32、FP16、BF16 的向量点积、矩阵乘、批量矩阵乘、批次广播、非连续输入、
零 K 和空输出；前向及左右输入一阶梯度分别验证。广播后的梯度要归约回原输入
形状。高阶梯度不在当前范围。精确输入规格见
[`goldendata/inputs_spec.yaml`](../goldendata/inputs_spec.yaml)。

### 2.2 参考与数值规格

[`reference.py`](../reference.py) 用 FP32 内部计算，结果转换为输入 dtype。
16 类形状与特殊值、三 dtype、每类三种子，共 144 组 CPU 黄金；每组同时
保存前向和双输入梯度。非零参考按 GEMM 相对误差 `5e-2`，零参考按 dtype
绝对容差判定。

## 3. 实现说明

### 3.1 路线选择

`torch.matmul` 是标准 ATen 入口，本期选择 A1 注册方式。昆仑芯 XMLIR
张量在 PyTorch 中表现为 CUDA，因此注册点是 `AutogradCUDA`；仅注册
`CUDA` 会使反向路径进入不适用的 `aten::matmul_backward`。生产计算使用
自研 Triton dot，`torch_level.py` 只作为语义组合对照。

### 3.2 kernel 与反向

[`kernel/_shape.py`](../kernel/_shape.py) 处理向量升降维、批次广播及梯度
归约。[`kernel/triton_level.py`](../kernel/triton_level.py) 默认用 32x32
输出 tile 的自研 `tl.dot` kernel；反向调用两次自研 matmul。外积/归约
已迁至 [`probes/outer.py`](../probes/outer.py)，通过 `matmul_outer(a, b)`
显式诊断；正式入口不再读取 `FLAGOS_MATMUL_TRITON_ALGORITHM`。

锁定 P800 栈中，BF16 `K=1` 的 dot 曾对部分梯度给出错误值。2026-10-14
通过可控 SRAM prime 和逐 pass IR，定位到 `TritonSDNNCombineBefore`
融合时丢失输入 tile 清零；K 被专门化为 1 时 MMA 仍计算 32 项。
现用 `@triton.jit(do_not_specialize=["K"])` 保留运行时 K，生成正确有效维
掩码，删除此前的 BF16 K=1 自动外积回退。专项 72/72、kernel 新增残留
回归 36/36、完整三级及自研/A1 黄金各 144/144 通过。
详见 [根因报告](k1_root_cause.md)。

`sum()` 反向可生成矩阵维零 stride 输入，之前的 dot 路径对此给出错误值；
进入 dot 前仍只在运行时识别为昆仑芯时实化这种矩阵维。其他厂商尚未验证。
`kernel/torch_level.py` 仅用于独立检查语义，不能记为自研 kernel。
本期没有厂商语言实现。

### 3.3 A1 注册与分发

[`register.py`](../register.py) 注册 `aten::matmul` 的 `AutogradCUDA`
实现并保留 `Library` 引用。P800 的张量经 XMLIR 表现为 CUDA；仅注册
`CUDA` 时，前向可命中但反向走到不适用的 `aten::matmul_backward`。
op 层同时验证 `torch.matmul`、显式 ATen 调用、CPU 不受影响和实际命中次数。

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

2026-10-23 将具体 backend 导入移到 `register_a1` 调用时，包导出扫描不再
要求 Triton。注册闭包绑定前后向实现，保留一阶 autograd 语义；缺少硬件
依赖时直接报告原异常，不再被脚本路径回退掩盖。包模式与脚本模式均有
CPU 回归，P800 验证范围与源码身份见 [PR 自查记录](pr-readiness.md)。

### 3.4 三级实现状态

| 级别 | 状态 | 说明 |
|---|---|---|
| torch 级 | ✅ | 形状整理、组合语义和 CPU 黄金对照 |
| Triton 级 | ✅ | 动态 K dot、前向、反向、广播梯度归约 |
| 硬件级 | ⬜ | 本期不开发厂商 SDK 或原生硬件语言版本 |

## 4. 验证结果

| 层级 | 锁定环境、物理 XPU 2 |
|---|---|
| CPU 参考 / `torch_level` 诊断 | 生成 144 组黄金；组合诊断对黄金 144/144 |
| Triton 黄金 / A1 黄金 | 各 144/144；A1 144 次命中 |
| kernel | 144 组、确定性/输入敏感哨兵、零 stride 梯度、36 次 SRAM 残留回归通过 |
| op | 15 次 A1 命中；排除 FlagGems `copy_` 后共存测试另有 15 次 |
| 小型应用 | FlagGems GELU + 两次 A1 matmul，通过且命中 2 次 |
| 真实 vLLM（历史探索） | 未形成模型对照；按 2026-10-14 范围调整，不作为本阶段验收项 |

kernel 最坏前向/反向相对误差分别为 `7.463e-3` / `7.752e-3`。
[`test-report.md`](test-report.md) 和 [`accuracy.md`](accuracy.md) 保留测试入口
及判卷结果。寒武纪、昇腾、壁仞尚未实测。

## 5. 性能

新增 `script/perf_common.py`，统一仓库性能 provider 和单算子性能 CLI 的
输入与计算口径。复用黄金生成器：12 个非空形状 × 3 dtype × 前后向 ×
自研/原生，共 144 条；各形状/dtype 取第一个黄金种子，保留非连续布局、
广播、零值和大值。空形状与零 K 保留精度测试。`gen_golden.py` 改用完整
模块导入，解决全局 provider 加载时与其他算子 `reference` 重名的问题。

XPU 2 首轮 144/144 成功并建立基线，原有其他算子 25 条未变。计时前
检查 CPU 参考精度，各用例独立进程、20 次预热、100 次调用及逐次标量
读回。反向单独计算 dA+dB，原生前向图准备不计时，自研不含 A1 包装。
自研前向约 0.091–0.153 ms，均慢于原生；反向约 0.178–0.341 ms。
完整对照和原生 autograd 成本说明见 [`performance.md`](performance.md)。

独立复测补齐到 144/144；完整门禁为 OK 136 / WARN 4 / FAIL 4，全部超阈值项来自
原生，自研 Triton 72 条均 OK。对 8 条原生异常各追加 3 次复核，结果见
`performance_xpu2_v1/anomaly_repeats.md`，暂不足以建立稳定回归结论。
XPU2 三种 dtype 的 A1 分发开销和新版反向性能入口已补测。
完整门禁不能判为通过。FlagGems `bmm` 历史自动调参失败，没有有效对照。

## 6. 已知问题与后续

| 问题 | 当前状态 |
|---|---|
| 正式物理 XPU 2 | 黄金、三级已通过；144 条性能基线和 144 条独立复测已完成 |
| 性能门禁 | OK 136 / WARN 4 / FAIL 4；8 条异常均为原生，已追加 3 次复核；不修改基线 |
| FlagGems `bmm` 自动调参异常 | 不把不可用对照写成通过 |
| BF16 `K=1` dot 大误差 | 已定位 SDNN combine-before 丢失清零；动态 K 修复通过，已撤掉自动回退；vendor 编译器源码补丁待上游处理 |
| tiny vLLM 历史环境错误 | 曾在点算子及 EngineCore 初始化阶段失败；本阶段只需小型 framework，停止该方向验证 |
| 其他三家国产硬件 | 各自环境、A1、精度、性能待验证 |
| 物理 XPU 1 的 PyTorch 标量读回超时 | 历史探索中故障在 matmul 前；三级入口先做限时设备探针 |

## 7. 结论与交付边界

昆仑芯正式物理 XPU 2 上，matmul 的 kernel、A1 op 和小型 framework 三层
均已通过；黄金、自研/A1 命中、K=1 修复和 SRAM 残留回归均有独立记录。
FlagGems matmul 对照因锁定版本接口/自动调参不可用而留档，不能把它记为
通过。性能门禁的 4 FAIL / 4 WARN 全部出现在原生路径，本阶段不做性能优化。

本报告对应的源码、测试入口和原始结果见 [总体报告](../REPORT.md)、
[测试报告](test-report.md)、[精度报告](accuracy.md) 和
[性能报告](performance.md)。

## 附录: 复现入口

在个人容器项目根目录，确认物理 XPU 2 空闲后执行：

```bash
python ops/matmul/script/gen_golden.py
python ops/matmul/script/check_accuracy.py --impl torch --device cpu
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul
```

当前 XPU 2 正式性能数据、源码哈希和门禁摘要位于
[`performance_xpu2_v1/`](performance_xpu2_v1/)。
