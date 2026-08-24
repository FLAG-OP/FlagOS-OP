# 样例: A1 × kernel 层

Triton kernel 直测——不经 aten 注册/分发，直接验证 kernel 本体:

1. 精度: 多 shape×dtype vs CPU 参考
2. 哨兵: 确定性（同输入两次调用一致）+ 输入敏感性
3. 性能: 短采样（规避分配器池增长失真）

```bash
python3 examples/a1-kernel/example.py
```
