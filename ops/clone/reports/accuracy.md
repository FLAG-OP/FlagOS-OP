# 精度分册: clone

## 1. 判卷标准

参考 [reference.py](../reference.py)：CPU/设备原生 `self.clone()`（纯拷贝，
位级标准）。容差按 dtype: fp32 1e-5，fp16/bf16 1e-2。

## 2. 黄金数据

- 规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **48** 组
- 特殊用例: zeros / large / boundary；非整倍数: 14336/5000/5120/4097

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
python3 script/check_accuracy.py --impl triton    --device mlu
```

## 3. 结果

| 实现 | 通过 | 最差 abs err |
|---|---|---|
| reference | 48/48 | 0 |
| torch 级 | 48/48 | 0 |
| Triton 级 | 48/48 | 0 |

## 4. 结构性检查（kernel 层）

- shape/dtype/stride 与原生 `preserve_format` 一致（含视图）。
- 存储独立: `out.data_ptr() != src.data_ptr()`；修改 out 不影响 src。
- 哨兵: 同输入两次一致、输入变化输出变化。
- 回退: int32 等走原生，与注册前原生参考一致。

## 5. 结论

精度全部位级一致，满足验收标准。
