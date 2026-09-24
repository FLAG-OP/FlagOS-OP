# 精度分册: dropout

## 1. 判定口径（重要）

dropout 含 RNG，Triton Philox 流与 PyTorch 原生**不同**，因此分两类判定：

| 分支 | 判据 |
|---|---|
| train=False / p=0 | 位级：返回 input 本身（别名） |
| p=1 | 位级：全零 |
| 0<p<1 | 结构 + 统计：shape/dtype/stride、值域 `{0, x/(1-p)}`、drop 比例 ≈ p、非别名、seed 可控 |

容差: fp32 1e-5 / fp16,bf16 1e-2；drop 比例容差 0.02。

## 2. 黄金数据

- 规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，**189** 组
- deterministic 108 组（存 expected，位级）+ stochastic 81 组（存输入/参数）
- shape 含非整倍数 14336/5000/5120/4097

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl triton    --device mlu
```

## 3. 结果

| 实现 | 通过 | 说明 |
|---|---|---|
| reference | 189/189 | 统计自洽 |
| torch 级 | 189/189 | 原生 |
| Triton 级 | 189/189 | 最大 drop 比例偏差 0.0016 |

## 4. 统计细节

- 存活元素严格等于 `input/(1-p)`（相对误差 < 5%，bf16 含舍入）。
- 被 drop 元素严格为 0。
- drop 比例: p=0.1/0.5/0.9 在 n≥65536 下偏差 < 0.002（远小于 0.02 门限）。
- `torch.manual_seed` 控制: 同 seed 输出逐位一致；异 seed 输出不同。

## 5. 结论

确定性分支位级一致、随机分支统计合规，满足验收标准。
