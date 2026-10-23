# 算子总体报告: matmul（`aten::matmul`）

| 项 | 值 |
|---|---|
| 算子 | `aten::matmul` |
| 实现路线 | **A1**：`AutogradCUDA` 接管 `torch.matmul` |
| 开发级别 | **Triton 级**（自研 `tl.dot`）；硬件级本期置空 |
| 目标平台 | Kunlunxin P800，正式物理 XPU 2 |
| 开发者 | 陶焱烽 |
| 环境 | 个人容器 `tyanfeng-flagos-p800-lock`；Torch 2.9.0+cu129；Triton 3.0.0+03c4c9be |
| 报告日期 / 状态 | 2026-10-23 / PARTIAL：已有昆仑芯功能与 C1 证据；性能门禁保留 4 FAIL / 4 WARN，尚未完成最终验收 |

本阶段按第一周期要求交付功能版本、正确性证据和初始性能记录。应用层以
小型 framework 消费者为准，真实模型接入和 tiny vLLM 不属于本阶段范围；
性能优化安排在后续周期。

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

当前 PR 自查和合并后回归状态见 [交付自查](reports/pr-readiness.md)。

## 1. 实现矩阵

| 层级 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 参考 | [reference.py](reference.py) | ✅ | FP32 内部计算，结果转换回输入 dtype |
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | 形状整理和组合语义对照，不计作自研 kernel |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ | 32×32 tile，自研 dot，支持动态 K |
| 形状与反向 | [kernel/_shape.py](kernel/_shape.py) | ✅ | 向量升降维、批次广播和梯度归约 |
| A1 注册 | [register.py](register.py) | ✅ | `aten::matmul` + `AutogradCUDA`，保留 Library 引用 |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ | 本期不写厂商硬件语言实现 |

正式路径统一使用 dot。BF16 `K=1` 通过
`do_not_specialize=["K"]` 规避昆仑芯后端的编译问题；外积/向量归约仅保留在
[`probes/outer.py`](probes/outer.py) 中作诊断，不参与生产分发。

## 2. 语义与范围

支持 FP32、FP16、BF16 的向量点积、矩阵乘、批量矩阵乘、批次广播、非连续
输入、零 K、空输出、零输入、大数和 `K=1`；前向及左右输入一阶梯度分别验证，
广播梯度归约回原输入形状。高阶梯度不在本期范围。

参考输入规格见 [`goldendata/inputs_spec.yaml`](goldendata/inputs_spec.yaml)。
CPU reference 采用 FP32 内部计算，GEMM 非零结果按相对误差 `5e-2` 判卷，
零参考按 dtype 绝对容差判卷。

## 3. 昆仑芯验证矩阵

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 直测 | `test/kernel_level.py` | ✅ 144 组、确定性/输入敏感哨兵、零 stride 梯度、36 次 SRAM 残留回归 |
| 黄金回归 | `script/check_accuracy.py --impl triton` | ✅ 144/144，最坏相对误差 `7.75e-3` |
| A1 黄金 | `--impl a1-triton` | ✅ 144/144，144 次命中 |
| A1 op | `test/op_level.py` | ✅ `AutogradCUDA` 15 次命中，CPU 隔离、显式 ATen 和单侧梯度通过 |
| FlagGems 共存 | `test/op_level.py --with-flaggems` | ✅ 排除既有 `copy_` 问题后 15 次命中 |
| 小型 framework | `test/framework_level.py` | ✅ FlagGems GELU + 两次 matmul；输出、梯度和 2 次命中通过 |
| 一键三级 | `example.py p800-kunlunxin` | ✅ kernel / op / framework 独立进程通过 |
| BF16 `K=1` 专项 | `probes/k1_lowering.py` | ✅ 动态 K 72/72；SRAM 残留回归 36/36 |

应用层不使用真实模型；物理 XPU 1 曾在 matmul 前的最小标量读回阶段超时，
不列为已验证设备。寒武纪、昇腾和壁仞尚未进行 matmul 验证。

### 应用消费者证据（历史验证：PASS）

| 字段 | 值 |
|---|---|
| consumer_type | `MatmulConsumer`：两次 matmul + FlagGems GELU，含一阶反向 |
| 证据等级 | C1 最小消费者 |
| 引擎 / 版本 | PyTorch eager 2.9.0+cu129；FlagGems 4.2.1rc0 |
| profile | `p800-kunlunxin`，物理 XPU 2 |
| 测试日期 | 2026-10-14；本次新快照回归单列于交付自查 |
| commit / 源码身份 | 历史结果采集时工作树未提交；对应代码已收录于 `f697348`，实测源码以 [SHA256 记录](reports/performance_xpu2_v1/provenance.json) 为准，不声称当时直接测试了该 commit |
| 可支持结论 | 此消费者输出、梯度及两次 A1 命中通过；不代表 C2 真实引擎验证 |

## 4. 关键问题与修复

BF16 `K=1` 错误来自昆仑芯 `TritonSDNNCombineBefore` 融合：K 被专门化后，
输入 tile 清零语义丢失，而 MMA 仍按 K=32 计算，读取了 SRAM 残留。通过
保留运行时 K 掩码修复，未修改共享编译器、FlagGems 或服务器配置。完整证据见
[K=1 根因报告](reports/k1_root_cause.md)。

锁定版 FlagGems 没有完整 matmul 入口，`bmm` 自动调参失败；原因、版本和
不可用结果已记录，不把缺失对照记成通过。高阶梯度、真实 vLLM、厂商硬件
语言实现和其他平台属于本阶段边界外或后续工作。

## 5. 性能记录状态

XPU 2 建立了 12 个非空形状 × 3 dtype × 前后向 × 自研/原生的 144 条基线，
并完成 144 条独立复测。自研 Triton 72 条均为 OK；完整门禁为 `OK 136 /
WARN 4 / FAIL 4`，8 条异常均来自原生路径，追加复核显示运行间波动。
当前不进行性能优化；完整数据见 [性能报告](reports/performance.md) 和
[门禁记录](reports/performance_xpu2_v1/gate.md)。

## 6. 交付物

| 交付物 | 位置 |
|---|---|
| 需求与范围 | [requirement.md](requirement.md) |
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| K=1 根因与修复 | [reports/k1_root_cause.md](reports/k1_root_cause.md) |
| XPU 2 性能记录 | [reports/performance_xpu2_v1/](reports/performance_xpu2_v1/) |
