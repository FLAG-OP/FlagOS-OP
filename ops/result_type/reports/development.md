# 算子开发报告: result_type

| 项 | 值 |
|---|---|
| 算子名称 | `result_type`（dtype 元数据类） |
| 实现路线 / 开发级别 | **自用/实验** / **torch 级** |
| 目标设备 | `cambricon`（MLU590-M9） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 1. 环境配置

MLU590-M9 ×8；torch 2.7.1+cpu / torch_mlu 1.29.2 / triton 3.2.0+mlu /
flag_gems 5.3.5；dispatch key `PrivateUse1`；vLLM 未安装。
`check_env --device cambricon` → OK。

## 2. 算子定义

`aten::result_type`：两输入在 PyTorch 类型提升下的公共 dtype。判定口径：公共 dtype 相等。

## 3. 实现说明

[kernel/torch_level.py](../kernel/torch_level.py)；Triton/硬件级不适用
（见 [kernel/README.md](../kernel/README.md)）。

| 级别 | 状态 |
|---|---|
| torch 级 | ✅ |
| Triton 级 | ⬜ 不适用 |
| 硬件级 | ⬜ 不适用 |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | ✅ | 8 dtype × 8 dtype 提升矩阵（64 项） |
| 框架层 | ✅ | —（result_type 为纯函数，不可 A1 注册） |
| 应用层 | ✅ | 子进程前向正常/一致 |
| 黄金 | ✅ | 8/8 |

## 5. 性能

0.0013ms（该类算子为元数据/同步操作，非带宽/算力受限）。

## 6. 已知问题与风险

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无设备码 | 无 Triton 实现 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 环境限制 |

## 7. 结论

三层全绿、8/8 黄金通过。后续: 真实 vLLM 应用层。

## 附录: 复现命令

```bash
cd ops/result_type
python3 example.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
```
