# P800 BF16 K=1 dot 根因与修复

日期：2026-10-14。设备为物理 XPU 2（`CUDA_VISIBLE_DEVICES=1,2`、逻辑
`cuda:1`），个人容器 `tyanfeng-flagos-p800-lock`；PyTorch `2.9.0+cu129`、
Triton `3.0.0+03c4c9be`。源码、测试和结果均位于个人工作区。

## 1. 结论

默认 BF16 编译路径在 **`TritonSDNNCombineBefore`
（`tritonsdnn-combine-before`）融合阶段丢失了输入 tile 的清零语义**。
K 被专门化为常量 1 时，输入 DMA 只写一个有效 K 元素，DS/MMA 却按 32
个元素计算。剩余 31 项可能来自上一次计算的片上 SRAM，造成错误结果。
这解释了此前相同输入、甚至相同编译产物有时正常、有时错误的现象。

本算子使用 `@triton.jit(do_not_specialize=["K"])` 保留运行时 K，
让后端产生正确的 DS/MMA 有效 K 掩码。已删除 BF16 K=1 的自动外积回退；
默认前向、反向统一使用原有 dot kernel。没有修改安装的编译器、共享
FlagGems 或服务器配置。外积/归约实现随后迁至
[`probes/outer.py`](../probes/outer.py)，通过显式 `matmul_outer(a, b)` 作诊断；
正式入口已移除算法环境变量切换。本文保留修复验证中的关键观测和复现方法。

## 2. FLAG-OP 中找到的相关信息

`https://github.com/FLAG-OP` 是组织主页。检查其项目和相关 issue 后，
最相关的是 [mmBF16FIX](https://github.com/FLAG-OP/mmBF16FIX)，
本轮阅读版本为 `09967ec546277c8175fe4d1021daeb3e3b3678cd`。

该项目的 [REPORT.md](https://github.com/FLAG-OP/mmBF16FIX/blob/09967ec546277c8175fe4d1021daeb3e3b3678cd/REPORT.md)
指出：`XMLIR_MATMUL_FAST_MODE=0` 将 BF16 dot 转成 FP32 SDNN 路径，
导致性能接近 FP32；设置为 1 会选择另一条 16-bit SDNN 路径。
这是性能问题的解释，没有直接给出本次 K=1 残留数据问题的根因或修复。
本轮据此对照两条 lowering 路径，并进一步做 SRAM 污染和逐 pass 分析。

## 3. 可控复现

[`probes/k1_lowering.py`](../probes/k1_lowering.py) 直接调用 dot，
绕过历史生产外积回退。先分配全部张量并预热，再紧接着执行：

1. FP32 `(128,32) @ (32,128)`，两个输入填相同的 `prime` 值；
2. BF16 `(37,1) @ (1,65)`，两个输入均填 0 或均填 1；
3. 读回完整的 37×65 输出，与数学结果比较。

步骤 1 与 2 之间没有额外设备张量计算。原始 K 专门化、fast mode 0：

| FP32 prime 值 | BF16 K=1 输入值 | 正确输出 | 实际输出（全部 2405 元素） |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 31 |
| 1 | 1 | 1 | 32 |
| 3 | 0 | 0 | 278 |
| 3 | 1 | 1 | 280 |

`31 = (32−1)×1×1`，精确对应被掩码排除的 31 个 K 项。
prime=3 时，额外和为 `31×3×3=279`，转成 BF16 后读回 278；再加有效项
1 后为 280。相同过程中 K=2 对照均正常；再次 prime=0 后 K=1 恢复正常。
改变逻辑输入外、已分配的 guard storage 数值也未改变正常结果。

| 编译设置 | 72 次检查中的失败次数 | 说明 |
|---|---:|---|
| 原始 K 专门化，fast mode 0 | 18 | 仅 K=1 且非零 FP32 prime 时失败 |
| 动态 K，fast mode 0 | 0 | 本次选用的算子级修复 |
| 原始 K 专门化，fast mode 1 | 0 | 相关仓库建议的另一编译路径，仅作对照 |

72 次包括 60 次前置计算对照及 12 次 guard 对照。三组使用不同进程和
独立新缓存，输入、tile 和设备相同。

## 4. 定位到具体编译步骤

用 `MLIR_ENABLE_DUMP=1` 和独立缓存编译原始 K=1 kernel，逐 pass 检查
生成的 IR。

**Triton 前端正确保留语义。** TTIR 中两次 `tt.load` 都带 K 掩码和
`other=0`；随后进行 BF16 `tt.dot`、FP32 累加。

**`TritonSDNNLegalize` 之后仍正确。** 对每个输入先分配 32×32 BF16 SRAM，
`triton_sdnn.ew fill` 清零整个 tile，再仅搬入有效的 `Mvalid×1` 或
`1×Nvalid` 区域，最后把整个 tile 转成 FP32。此时 MMA 计算完整 K=32
是合理的，因为无效项已经清零。

**`TritonSDNNCombineBefore` 之后清零消失。** 原来的“清零 + BF16 DMA +
整块转 FP32”融合成直接向 FP32 SRAM 搬运；只写有效子区域，未保留整块
清零。DS 仍为 `[32,32]`，MMA 仍为 `[32,32,32]`。
该缺陷随后进入最终 TTXIR。后续 LLVM lowering 虽包含 BF16 扩展所需的
DMA 清零/转换指令，但从实际输出可知，它没有补足无效 K 元素的零值语义。

因此问题已经定位到 **TTIR→SDNN lowering 内的 DMA/类型转换融合步骤**，
并有真实执行的残留值对照支持。尚未取得该 vendor pass 的 C++ 实现源码，
不能给出其内部具体 rewrite pattern 或源文件行号，也未制作编译器源码补丁。

另做过进程内跳过整个 `add_tritonsdnn_combine_before_pass` 的诊断：
编译后启动原始 kernel 报 `status=700` / `err_code: -714`，未进入 72 项
判卷，因此**不能把整步禁用作为修复，也不能将此实验记为通过**。
该临时 monkeypatch 仅存在于诊断进程，未修改安装文件；退出后使用正常
编译流程重跑 kernel，36 次残留回归、144 组和哨兵全部通过。

## 5. 修复选择与回归

在 [`kernel/triton_level.py`](../kernel/triton_level.py) 中禁止 K 专门化后，
K=1 不再被前端折叠掉；最终 TTXIR 的 DS 掩码为 `[32,Kvalid]` 和
`[Kvalid,32]`，MMA 为 `[32,32,Kvalid]`，本例 Kvalid=1。
因此无效的 31 个 K 项不再参与计算。数学、广播和反向公式没有改变。

该选项也应用于其他 dtype/K，避免新增第二份 dot kernel；其他厂商仍需
分别测试。保留已有的昆仑芯矩阵维零 stride 实化，与本次修复分开处理。
不默认设置 `XMLIR_MATMUL_FAST_MODE`：它是进程级编译选项，会影响其他
算子；本次无需依靠它即可修复。对照不同 fast mode 必须使用独立缓存，
因为安装版 XPUOptions 的哈希未直接包含这一环境变量。

在 fast mode 0、独立新缓存中完成以下回归：

| 检查 | 修复后结果 |
|---|---|
| 新增 FP32 prime→BF16 K=1/2 kernel 回归 | 36/36，通过；检查完整输出精确等于 0/1/2 |
| kernel 全部形状/dtype/前向/反向 | 144 组 + 哨兵 + 零 stride，通过 |
| 直调 Triton 黄金 | 144/144，最坏相对误差 `7.75e-3` |
| A1 黄金 | 144/144，144 次命中，最坏相对误差 `7.75e-3` |
| op | 15 次命中，通过 |
| FlagGems 共存 op | 排除既有 `copy_` 问题后 15 次命中，通过 |
| framework | GELU + 两次 A1 matmul，前向/梯度通过，2 次命中 |

修复后的自研 BF16 基准：`(B,M,K,N)=(1,63,127,65)` 为 `0.101620 ms`，
`(1,128,256,192)` 为 `0.099555 ms`，`(1,37,1,65)` 为 `0.105918 ms`。
前两项与此前 `0.097733/0.096940 ms` 同量级，单轮采样不足以宣称加速
或稳定性能持平。使用相同 CPU 种子、独立进程、20 次预热、100 次读回采样。
上述修复专项计时当时未更新正式性能基线；后续完整采集和门禁以
[当前性能报告](performance.md) 为准。其他国产平台验收仍待完成。后续范围已调整为小型
framework，真实模型及 tiny vLLM 不列入本阶段待办。

## 6. 复现命令

在个人容器项目根目录、确认物理 XPU 2 空闲后运行。每种设置使用新目录：

```bash
# 原始专门化路径：预期返回 1，报告错误（这是复现成功）。
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 \
TRITON_CACHE_DIR="/tmp/matmul_k1_repro_original_cache" \
python ops/matmul/probes/k1_lowering.py --prime-dtype float32 --repeats 3 \
  --variant original --output /tmp/matmul_k1_repro_original

# 修复后的生产 dot：预期返回 0。
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 \
TRITON_CACHE_DIR="/tmp/matmul_k1_repro_dynamic_cache" \
python ops/matmul/probes/k1_lowering.py --prime-dtype float32 --repeats 3 \
  --variant dynamic --output /tmp/matmul_k1_repro_dynamic

CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 \
python ops/matmul/example.py p800-kunlunxin
```

`original` 会在探针中显式重建允许专门化的旧 kernel，生产实现修复后仍可
用于复现。 dynamic 使用当前生产 dot。需要查看 lowering 时，可设置
MLIR_ENABLE_DUMP=1 并将 stderr 重定向到临时目录。
