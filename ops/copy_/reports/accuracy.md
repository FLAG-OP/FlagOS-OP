# 精度分册: copy_

## 1. 判卷标准

参考 [reference.py](../reference.py)：原生 `dst.copy_(src)`（纯搬运）。
容差按 **dst dtype**: fp32 1e-5 / fp16,bf16 1e-2。判定含值、原地返回
（`out is dst`）、dtype。boundary 溢出到 ±inf 用 isclose（inf==inf）。

## 2. 黄金数据

- 规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，**78** 组
- 三类: `same`（同形同/异 dtype）、`cast`（异 dtype）、`broadcast`（形状不同）
- 非整倍数: 14336/5000/4097；特殊: boundary/zeros/large

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
python3 script/check_accuracy.py --impl triton    --device mlu
```

## 3. 结果

| 实现 | 通过 | 最差 abs err |
|---|---|---|
| reference | 78/78 | 0 |
| torch 级 | 78/78 | 0 |
| Triton 级 | 78/78 | 0 |

## 4. 结构性检查（kernel 层）

- 原地返回 `out is dst`，dtype 恒为 dst.dtype。
- 广播 `[64,128,256]<-[128,1]`、`[8,1,5000]<-[8,1,1]`、`[16,32]<-[1,32]` 全对。
- 非连续 dst（`[::2,1::3]`）正确。
- 自拷贝（`dst.copy_(dst)`）走回退，不崩且不变。
- 哨兵: 同输入两次一致、输入变化输出变化。

## 5. 结论

精度与结构均满足验收标准。
