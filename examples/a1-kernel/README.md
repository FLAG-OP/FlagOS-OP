# 样例: A1 × kernel 层

Triton kernel 直测——不经 aten 注册/分发，直接验证 kernel 本体:

1. 精度: 多 shape×dtype vs CPU 参考
2. 哨兵: 确定性（同输入两次调用一致）+ 输入敏感性
3. 性能: 短采样（规避分配器池增长失真）

```bash
python3 examples/a1-kernel/example.py
```

## 同算子贯穿三层

本路线三层样例各自独立可跑；若要跟踪**同一算子**走完
kernel→op→framework，把各层样例的算子替换为同一个即可。
gelu_and_mul 的贯穿实证见 `tests/kernel_level/test_consistency.py`
与旗舰样例 [examples/b-fullstack](../b-fullstack/)。
