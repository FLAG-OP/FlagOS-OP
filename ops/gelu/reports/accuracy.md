# 精度分册: gelu（2026-11-25/26）

## 容差依据（docs/acceptance.md 唯一权威表）

- pointwise 有界输出 → abs: fp32=1e-5(2e-5) / fp16|bf16=1e-2
- GEMM/乘加放大形态 → rel: 5e-2（除以 max|ref|）

## 黄金校验（真机 cuda:6）

```
triton: [gelu:triton@cuda:6] 150/150 PASS, worst=9.766e-04；native: [gelu:native@cuda:6] 150/150 PASS, worst=9.766e-04
```

## 误差来源说明

- bf16 量化噪声 ~1e-2（容差档依据）; 参考实现 fp32 内部 + cast 回,
  不把量化噪声当实现误差;
- 累加型算子（无（逐 elementwise））以独立高精度参考判卷, 消除累加序噪声。

## 边界记录

- 非有限值 parity 以官方 CPU ATen 为锚（+inf 行为的平台分化见 PLATFORM.md）。
