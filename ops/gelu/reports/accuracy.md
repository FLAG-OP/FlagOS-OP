# 精度分册: gelu（2026-11-25/27）

## 容差依据（docs/acceptance.md 唯一权威表）

- pointwise 有界输出 → abs: fp32=1e-5(2e-5) / fp16|bf16=1e-2
- GEMM/乘加放大形态 → rel: 5e-2（除以 max|ref|）
- 参考改为独立 fp64；FlagFix 负尾反例用
  `python3 script/check_accuracy.py --impl triton --device cuda:6 --flagfix-replay`
  逐项重放，`none/tanh × forward/backward` 分别记 PASS/FAIL。

## FlagFix 负尾重放（真机 cuda:6, 2026-11-27）

```
none/forward:   PASS err=3.730008302e-09
none/backward:  PASS err=6.138343167e-10
tanh/forward:   PASS err=4.160485151e-11
tanh/backward:  PASS err=3.884027822e-10
[gelu:triton@cuda:6] flagfix 4/4 PASS
```

判定使用独立 fp64 参考；`+inf` 仍作为 observed，不计 PASS。

## 黄金校验（真机 cuda:6）

```
triton: [gelu:triton@cuda:6] 150/150 PASS, worst=9.766e-04；native: [gelu:native@cuda:6] 150/150 PASS, worst=9.766e-04
```

## 误差来源说明

- bf16 量化噪声 ~1e-2（容差档依据）; 参考实现 fp64 内部 + cast 回,
  不把量化噪声当实现误差;
- 累加型算子（无（逐 elementwise））以独立高精度参考判卷, 消除累加序噪声。

## 边界记录

- 非有限值 parity 以官方 CPU ATen 为锚（+inf 行为的平台分化见 PLATFORM.md）。
