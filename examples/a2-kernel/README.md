# 样例: A2 × kernel 层

Triton gelu_and_mul kernel 直测（不经 dispatch），与 A1-kernel 同构。

```bash
python3 examples/a2-kernel/example.py
```

## 同算子贯穿三层

本路线三层样例各自独立可跑；若要跟踪**同一算子**走完
kernel→op→framework，把各层样例的算子替换为同一个即可。
gelu_and_mul 的贯穿实证见 `tests/kernel_level/test_consistency.py`
与旗舰样例 [examples/b-fullstack](../b-fullstack/)。
