# 精度分册: type_as

## 1. 判卷标准

参考实现 [reference.py](../reference.py)：`self.to(other.dtype)`，在
**CPU** 上计算（cast 无算术放大，CPU 即最可信）。容差按**输出 dtype**
分档（[验收标准](../../../docs/acceptance.md#accuracy)）：

| 输出 dtype | abs 容差 |
|---|---|
| float32 | 1e-5 |
| float16 | 1e-2 |
| bfloat16 | 1e-2 |

## 2. 黄金数据

- 规格: [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)
- 组数: **111**（3 个 case × 多 shape × self/target dtype 交叉 × 3 seed）
- 特殊用例: `zeros`（全零行）、`large`（±1e4）、`boundary`（dtype 上限/2，
  含宽→窄溢出到 inf 的合法语义）
- 非 BLOCK 整倍数维度: 14336 / 5000 / 5120 / 4097

复现：

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
python3 script/check_accuracy.py --impl triton    --device mlu
```

## 3. 结果

| 实现 | 通过 | 最差 abs err |
|---|---|---|
| reference（CPU） | 111/111 | 0 |
| torch 级（MLU） | 111/111 | 0 |
| Triton 级（MLU） | 111/111 | 0 |
| FlagGems `ops.to_copy`（MLU，抽样 4 组） | 4/4 | 0 |

cast 为确定性 round-to-nearest，三端实现位级一致。

## 4. 结构性精度检查（kernel 层）

- 非 dense 视图（转置 / permute / 步长切片 / expand / channels_last）:
  err 0，且输出 stride 与 ATen `preserve_format` 一致。
- 同 dtype 别名快路径: `type_as_triton(x, x) is x`。
- 回退: int32 等不支持 dtype 走原生，结果与参考一致。
- 哨兵: 同输入两次一致；输入变化输出变化（`x+1`）。

## 5. 结论

精度全部位级一致，满足验收标准。
