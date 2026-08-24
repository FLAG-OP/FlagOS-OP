# 算子层测试

每路线一个测试模块，统一签名 `run(profile) -> bool`，无任何硬编码设备。

标准三段式:

| 阶段 | 验证 | 断言 |
|---|---|---|
| ① 精度 | 实现 vs PyTorch 参考 | 多 shape×dtype 误差 < 容差 |
| ② 分发 | 注册数、policy 切换、fallback | impl 被正确选中 |
| ③ 性能 | latency / 有效带宽 / 加速比 | 记录基线（CI 可加阈值） |

容差: fp32 1e-5, bf16/fp16 1e-2（先 .float() 再比）

```bash
python3 run.py --route a2 --level op --device <profile名>
python3 tests/op_level/test_a2.py <profile名>   # 直接调用
```
