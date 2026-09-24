# 精度报告: sdpa（Ascend 910，2026-09-16 实测）

## 1. 用例覆盖

### kernel 层直测（36 组）

| 类别 | shape | 说明 |
|---|---|---|
| basic | B1 H4/4 S128 D64 | 基线 |
| causal | 同上 + is_causal | 下三角 |
| batch | B2 H8/8 S256 D64 | 批量 |
| tail100 / tail333 | S=100 / 333 | **非 BLOCK 整倍数**（#15a 固化） |
| vit-d80 | S197 D80 | D 非 2 幂 |
| gqa / gqa-boolmask | H8/2 S256 | GQA 广播 + mask 组合 |
| floatmask | S128 | 加性 mask |
| d128 | B2 S512 D128 | 大 head_dim |
| decode-sq1 | S1/7 | decode 形态 |

dtype: fp32 / fp16 / bf16。哨兵: 确定性 + 输入敏感（抓静默 no-op 类）。

### 黄金（265 组，CPU fp32 生成）

basic（2B×4H×S128×D64 × 3 dtype × causal/mask 组合 × 2 seed）·
tail（S100/197/333 × D64/80 × GQA）· extreme（scale=8 + zeros/large
行，fp32-only）。生成时**双向互验**（参考 vs CPU 官方实现，分歧即报错）。

## 2. 结果

| dtype | 自研 max err | 原生（对照） | 判定 |
|---|---|---|---|
| fp32 | 1.2e-7 | ~1e-7 | ✓✓ |
| fp16 | 9.8e-4 | ~4e-4 | ✓ |
| bf16 | 3.9e-3 | ~3.9e-3 | ✓ |

黄金 265/265 PASS（自研与原生双双全过，worst 见测试报告§3）。

## 3. 关键精度决策（均有实测依据）

| 决策 | 依据 |
|---|---|
| fp32 dot 用 `input_precision="ieee"` | 默认 tf32（10-bit 尾数）在 extreme 实测 3.381e-4 超容差；ieee 后通过 |
| extreme case 用相对容差 5e-3 | softmax 饱和下 fp32 累加顺序差异 ~3e-4 属正常——**原生实现与黄金同为 3.381e-4**，自研 vs 原生仅 3.8e-6（互咬合） |
| extreme 不配 fp16/bf16 | 输入量化翻转饱和 softmax argmax 是 dtype 固有性质（实测原生同误差 1.39e-1），非实现缺陷 |
| 全遮蔽行输出 NaN | 与 aten softmax(-inf 行)=nan 语义对齐（参考与 kernel 同路） |
| 测试输入 CPU 生成后搬 NPU | flag_gems enable 接管 rand 族且其 kernel 有 UB 溢出已知缺陷，避免生成路径污染被测结论 |

## 4. 失败分析（历史，均已闭环）

| 现象 | 根因 | 处置 |
|---|---|---|
| bool mask 组 err 4.8e-2 | **参考写错**: bool→float 把遮蔽语义变加性 | 参考修正后全过（kernel 无错） |
| extreme fp32 FAIL 3.4e-4 | tf32 dot 默认精度 | ieee 修复 |
| extreme fp16 FAIL 1.4e-1 | dtype 固有（原生同值） | case 改 fp32-only |
| 黄金互验断言三连报 | fp16 量化/fp32 累加序/饱和放大 | 阈值按 dtype 分档 + extreme 相对阈值 |

## 5. 复现

```bash
python3 test/kernel_level.py                                  # 36+哨兵
python3 script/gen_golden.py                                  # 265 组 CPU
python3 script/check_accuracy.py --impl reference --device cpu # 265/265
python3 script/check_accuracy.py --impl triton  --device npu:0 # 265/265
python3 script/check_accuracy.py --impl native  --device npu:0 # 原生对照
python3 test/op_level.py                                      # 拦截+梯度
```
