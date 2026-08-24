# 样例: A2 路线 × 算子层

## 目标

开发 vLLM 融合算子（非 aten 算子），通过 **FlagOS dispatch**
注册双后端并完成算子层三段式测试。

## 三步流程

1. Triton kernel + PyTorch 参考实现（先定语义再对齐）
2. 注册双后端: `default.flagos`(150) + `reference.torch`(50)
3. 三段式: 精度（多 shape×dtype）/ dispatch 策略切换 / 性能

## 运行

```bash
python3 examples/a2-op/example.py
```

## 预期输出

```
精度: 9/9 组合 PASS
dispatch: flagos→default.flagos / reference→reference.torch PASS
性能: Triton=0.05x ms (数千 GB/s)  参考=3xx ms  加速=数千x
```

## 关键点 / 坑

1. 插件入口函数必须是 `register` / `vllm_fl_register`
   （官方 README 的 `register_builtins` 仅适用内置 backend）
2. 每个 dispatch 算子应有 reference 后端兜底——也是精度基准
3. 同进程串跑多测试需 `reset_default_manager()`（OpManager 缓存）
4. 性能基准用短采样（100 次内），避免输出分配器池增长失真

## 复制为开发起点

替换 `gelu_and_mul_kernel` 与两个 backend 的 `op` 方法即可。

## 同算子贯穿三层

本路线三层样例各自独立可跑；若要跟踪**同一算子**走完
kernel→op→framework，把各层样例的算子替换为同一个即可。
gelu_and_mul 的贯穿实证见 `tests/kernel_level/test_consistency.py`
与旗舰样例 [examples/b-fullstack](../b-fullstack/)。
