# 算子测试报告: matmul

| 项 | 值 |
|---|---|
| 算子名称 | `aten::matmul` |
| 实现路线 / 开发级别 | A1 `AutogradCUDA` / Triton 自研 dot |
| 目标设备 | Kunlunxin P800，物理 XPU 2 |
| 测试日期 | 2026-10-14～2026-10-15；合并后功能复测 2026-10-23 |
| 环境 | 个人锁定容器；Python 3.10.18、Torch 2.9.0+cu129、Triton 3.0.0+03c4c9be |
| 设备 | P800 物理 XPU 2：`CUDA_VISIBLE_DEVICES=1,2`，逻辑 `cuda:1` |
| 结论 | PARTIAL：历史正式目标卡黄金、三级、FlagGems 共存测试通过；性能基线与独立复测均完成 144 条，门禁 FAIL 4 / WARN 4 |

2026-10-14 当前范围：应用层只需小型 framework，真实模型接入和 tiny
vLLM 不作为本阶段验收项。模型探索已归档，不再安排复测。

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

2026-10-23 新快照的逐项结果与源码身份见 [PR 自查记录](pr-readiness.md)。
下面保留历史完整验证表，性能数字仍对应原采集日期。

## 1. 测试范围与结果

| 层级 | 入口 | 本轮结果 |
|---|---|---|
| CPU 黄金生成 | `script/gen_golden.py` | 历史生成 144 组前向及双输入梯度参考；本轮校验每份 SHA256 |
| CPU `torch_level` 组合诊断 | `check_accuracy.py --impl torch --device cpu` | 历史对黄金 144/144；不计为自研实现 |
| kernel 黄金 | `check_accuracy.py --impl triton` | 144/144，前向与双输入梯度 |
| A1 黄金 | `check_accuracy.py --impl a1-triton` | 144/144，144 次命中 |
| kernel 直测 | `test/kernel_level.py` | 144 组、确定性/输入敏感哨兵、零 stride 反向、36 次 SRAM 残留回归通过 |
| A1 op | `test/op_level.py` | `AutogradCUDA` 15 次命中；direct ATen、CPU 隔离、单侧梯度通过 |
| FlagGems 共存 | `test/op_level.py --with-flaggems` | 排除 `copy_` 后 15 次命中，通过 |
| 应用层 | `test/framework_level.py` | FlagGems GELU + 两次 matmul；输出及三个梯度均通过，2 次命中 |
| 一键三级 | `example.py p800-kunlunxin` | kernel/op/framework 独立进程全部通过 |
| BF16 `K=1` dot 专项 | `probes/k1_lowering.py`，XPU 2 | 原始路径 SRAM prime 对照 18/72 失败；动态 K 修复后 72/72 通过，详见根因报告 |

kernel 最坏前向相对误差 `7.463e-3`，最坏反向 `7.752e-3`；op 层最坏设备原生相对误差 `1.087e-2`；应用层输出/梯度最大误差 `3.509e-3`。

本表黄金、三级及共存结果均已在 `do_not_specialize=["K"]` 修复并删除
BF16 K=1 自动回退后重跑，使用 fast mode 0 与独立新缓存。
编译 pass 定位和残留数据对照见 [k1_root_cause.md](k1_root_cause.md)。

## 2. 应用与环境限制

当前应用层按小型 framework 验收，输出、梯度及两次 A1 命中均已在 XPU 2 验证。
物理 XPU 1 历史上在最小标量读回阶段超时，因此不列为已验证设备。
历史个人镜像未预装 pytest，当时未运行仓库级 tests/unit。合并后的 CPU CI
检查与目标卡功能回归见 [交付自查](pr-readiness.md)；历史结果不自动代表新快照通过。
C1 消费者的类型、版本、profile 和源码身份见 [总体报告](../REPORT.md)。

## 3. 性能状态

XPU 2 完整基线轮 144/144 成功且计时前精度检查通过：12 个非空形状 × 3 dtype × 2 方向 × 自研/原生。采用黄金生成器的固定 CPU 输入，空形状和零 K 仍只用于精度测试。基线已合并，其他算子原有 25 条未变。

独立复测补齐到 144/144。完整结果为 OK 136 / WARN 4 / FAIL 4，超阈值项全为原生，自研 Triton 72 条均 OK。追加复核仍显示原生延迟有较大波动，**完整性能门禁未通过**。XPU2 三种 dtype 的 A1 分发开销和新版反向性能入口已补测。全部证据见 [performance.md](performance.md) 和 [performance_xpu2_v1/](performance_xpu2_v1/)。

## 4. 复现

从个人容器项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/test/op_level.py p800-kunlunxin --with-flaggems
```

执行前先确认物理 XPU 2 空闲。

## 5. 诊断 kernel 迁移回归（2026-10-14）

将外积和向量归约 kernel 原样迁至 `probes/outer.py`，正式入口只保留 dot，
移除旧算法环境变量切换。`k1_dot.py` 通过显式 `matmul_outer` 对照，
`k1_lowering.py` 从新位置导入外积 kernel。

在同一个人容器、物理 XPU 2、fast mode 0 下验证：

- 诊断入口 18 项通过：FP32/FP16/BF16 各测试 K=1 外积、单行、单列、
  批次广播、向量点积、零 K；编译探针的外积入口另作一次全 1 检查并通过。
- `example.py p800-kunlunxin` 三级通过：kernel 144 组及 36 次 SRAM 残留
  回归、op 15 次命中、framework 2 次命中。
- `k1_dot.py --dtype bfloat16 --repeats 1` 对照完成，黄金 6 组、K 补零及
  tile 边界检查均无超容差结果。

本次为目录与诊断入口调整，未重新采集性能；根因报告保留了迁移前的
验证结论和复现入口。

## 6. 问题与风险

| # | 项目 | 影响 | 当前处理 |
|---|---|---|---|
| 1 | 性能门禁 `FAIL 4 / WARN 4` | 8 条原生路径超出本轮基线阈值 | 追加 3 次复核后仍保留原始判定；本阶段不进行性能优化 |
| 2 | FlagGems `bmm` 自动调参失败 | 无有效 matmul 三方性能数字 | 记录版本和具体错误；不把不可用路径记为通过 |
| 3 | 仓库 CPU CI 与目标硬件验证分开 | 镜像是否预装 pytest 不决定 CI 是否通过 | 合并后的检查见交付自查；不在厂商 Python 环境覆盖安装依赖 |
| 4 | 其他国产平台未验证 | 不能把 P800 结果外推到其他平台 | 后续按各自 profile 独立验证 |
| 5 | 高阶梯度、真实 vLLM | 不在当前第一阶段范围 | 明确列为边界，不作为本阶段阻塞 |

## 7. 总结

昆仑芯 XPU 2 上，matmul 的 kernel、A1 op 和小型 framework 三层测试全部
通过；黄金 144/144、A1 命中 144 次、K=1 动态 K 专项 72/72、SRAM
残留回归 36/36。测试覆盖三种 dtype、向量/矩阵/批量/广播/非连续布局、
前向和左右输入一阶反向。性能记录及遗留门禁状态见
[性能报告](performance.md)，不影响本阶段功能结论。
