# 精度报告: matmul

| 项 | 值 |
|---|---|
| 算子 / 路线 | `aten::matmul` / A1 `AutogradCUDA` |
| 目标设备 | Kunlunxin P800，物理 XPU 2 |
| 环境 | `tyanfeng-flagos-p800-lock`；Torch 2.9.0+cu129；Triton 3.0.0+03c4c9be |
| 报告状态 | PASS（昆仑芯已测精度范围）；FlagGems matmul 对照不可用原因已记录 |

## 1. 用例与判卷

`goldendata/inputs_spec.yaml` 共 16 类输入 × FP32/FP16/BF16 × 三种子 = 144 组。覆盖向量、矩阵、批量广播、非连续布局、尾块、零 K/空输出/空批次、零输入、大数、2×2 和 `K=1`；每组保存前向及左右输入梯度。CPU reference 用 FP32 内部计算后转回输入 dtype。非零参考按 GEMM 相对误差 `5e-2` 判定，全零参考使用 dtype 绝对容差。

## 2. 锁定环境结果

个人容器 `tyanfeng-flagos-p800-lock`，PyTorch `2.9.0+cu129`，Triton `3.0.0+03c4c9be`。正式映射 `CUDA_VISIBLE_DEVICES=1,2`，逻辑 `cuda:1` 为物理 XPU 2；下表除注明外均为该卡结果。

| 路径 | 前向 + dA + dB | A1 命中 | 结论 |
|---|---:|---:|---|
| CPU `torch_level` 组合诊断 | 144/144 | 不适用 | 旧有结果；不计为自研实现 |
| 自研 Triton dot（动态 K 修复后） | 144/144 | 不适用 | 通过，最坏 `7.75e-3` |
| A1 自研 Triton | 144/144 | 144 | 通过，最坏 `7.75e-3` |
| kernel 直测 | 144 组 + 哨兵 + 零 stride 梯度 + 36 次 SRAM 残留回归 | 不适用 | 通过 |
| A1 op 对照设备原生 | 15 次 | 15 | 通过，最坏相对误差 `1.087e-2` |
| A1 与 FlagGems 共存 | 15 次 | 15 | 排除其 `copy_` 后通过 |

CPU 黄金由 `reference.py` 调用 CPU PyTorch 生成，用作设备实现的判卷依据；
表中 CPU `torch_level` 一行另外检查形状整理和 `bmm` 组合逻辑，
它不是自研 Triton 精度的替代证明。

首次完整 dot 回归为 133/135：BF16 `matrix_vector` 的两组左梯度失败。查明其反向会生成 `K=1` 矩阵乘；该形态改用自研外积/归约后，追加直接的 `rank_one` 用例，达到 144/144。随后 op 层发现 `sum()` 的零 stride 梯度使 dot 双输入梯度错误；矩阵维零 stride 进入 dot 前实化。两项均纳入 kernel 直测和黄金回归。

在 XPU 2 用 [`probes/k1_dot.py`](../probes/k1_dot.py) 绕开回退，复现 BF16 `rank_one`/`matrix_vector` 左梯度大误差；K 补零到 2、4、8 等长度均正常。早期冷缓存与旧缓存实验时好时坏，仅凭编译产物差异不能确定原因。

进一步用 [`probes/k1_lowering.py`](../probes/k1_lowering.py) 控制前一条 kernel：FP32 全 1 的 K=32 matmul 后，BF16 K=1 全零输入得到 31、全 1 输入得到 32。逐 pass IR 显示 `TritonSDNNCombineBefore` 在 BF16→FP32 融合时删掉 tile 清零，而 MMA 仍计算 K=32；结果取决于 SRAM 残留。原始路径 72 次中 18 次失败，禁止 K 专门化后 72 次全通过。

生产 dot 已加 `do_not_specialize=["K"]` 并撤掉自动外积回退；fast mode 0 新缓存下重跑黄金各 144/144、三级和 FlagGems 共存均通过，新加 kernel 残留回归 36/36。完整证据、设置和复现命令见 [K=1 根因报告](k1_root_cause.md)。其他平台待验证。

## 3. 关键精度决策

| 决策 | 依据与结果 |
|---|---|
| FP32 内部参考后 cast 回输入 dtype | 避免把 BF16/FP16 量化噪声误判为实现误差 |
| GEMM 使用相对误差 `5e-2` | 输出随输入放大，绝对误差不适合作为统一判据 |
| `K=1` 保留运行时 K | 修复 P800 `TritonSDNNCombineBefore` 丢失 tile 清零的问题 |
| 矩阵维零 stride 在昆仑芯 dot 前实化 | 避免 `sum()` 反向的全 1 梯度触发错误读法 |
| 哨兵与残留回归 | 相同输入确定性、输入变化敏感；SRAM 残留专项 36/36 通过 |

## 4. 对照边界

锁定版 FlagGems `bmm` 在自动调参阶段报错，无法形成有效精度/性能对照。寒武纪、昇腾和壁仞尚未验证。

复现：`script/gen_golden.py`、`script/check_accuracy.py --impl triton --device p800-kunlunxin`、`--impl a1-triton`；执行前设置本轮设备映射。环境和命令见 [开发报告](development.md)。

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

2026-10-23 在相同 4.2 环境锁、物理 XPU 2 上重建 CPU 黄金并复测：
直调 Triton、A1 各 144/144，A1 命中 144 次，最坏相对误差均为 `7.75e-3`。
kernel 的 144 组、哨兵、零 stride 反向及 K=1/2 残留回归 36/36 通过。
该结果针对注册延迟导入改动后的快照，源码身份见 [PR 自查记录](pr-readiness.md)。
