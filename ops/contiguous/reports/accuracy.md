# 精度分册: contiguous

## 1. 判卷标准

参考 [reference.py](../reference.py)：原生 `.contiguous(memory_format)`。
容差 fp32 1e-5 / fp16,bf16 1e-2；额外判定连续性 + 别名语义。

## 2. 黄金数据

- 规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **114** 组
- 每组合两种 layout：`contig`（别名快路径）/ `view`（transpose，拷贝路径）
- 非整倍数: 14336/5000/5120/4097；特殊: zeros/large/boundary

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
python3 script/check_accuracy.py --impl triton    --device mlu
```

## 3. 结果

| 实现 | 通过 | 最差 abs err |
|---|---|---|
| reference | 114/114 | 0 |
| torch 级 | 114/114 | 0 |
| Triton 级 | 114/114 | 0 |

## 4. 结构性检查（kernel 层）

- 连续性: 所有拷贝路径输出 `is_contiguous() == True`。
- 别名语义: 连续输入返回 self 本身（`out is inp`）；非连续输入分配新存储。
- stride 与原生一致；转置/permute/步长/expand 全 err 0。
- 哨兵: 同输入两次一致、输入变化输出变化。

## 5. 结论

精度与结构均满足验收标准。
