# 样例: A1 路线 × 算子层

## 目标

从零写一个 Triton 算子（relu），通过 **torch aten dispatcher** 接管
`torch.relu`——任何框架代码（vLLM/transformers/自研脚本）调用
`torch.relu` 都自动走你的 kernel，无需改业务代码。

## 四步流程

1. 写 Triton kernel：`@pointwise_dynamic + @triton.jit`
2. aten 注册：`torch.library.Library("aten","IMPL").impl("relu", fn, key)`
3. 拦截验证：`torch.relu(x)` 命中计数器
4. 精度+性能：vs CPU 参考（CPU 张量不受设备 key 影响）

## 运行

```bash
python3 examples/a1-op/example.py                  # 默认参考 profile
python3 examples/a1-op/example.py <profile名>      # 切设备 profile
```

## 预期输出

```
拦截成功: torch.relu -> Triton kernel (calls=1)
精度: bf16/fp16/fp32 位级一致（relu 为精确算子）
性能(采样): Triton=0.047ms  CPU=110ms  加速=2000+x（绝对值视环境负载而定）
```

## 关键点 / 坑

1. dispatch key 来自设备 profile（P800 上 XPU 伪装 CUDA → `CUDA`）
2. `Library` 对象必须保持引用（否则注册被回收）
3. 同一 (算子, key) 重复注册会**覆盖**——与 `flag_gems.enable()`
   同进程使用时注意黑名单（框架级样例会演示）
4. CPU 张量不受影响，天然可用作精度参考

## 复制为开发起点

把 `relu_kernel` 换成你的算子逻辑、`"relu"` 换成目标 aten 算子名即可。

## 性能测量注意

逐次新分配输出张量的循环基准中，迭代次数过多（如 500 次 × 128MB 输出）
会触发分配器池增长，均值被抬高一两个数量级——本机实测同 kernel
100 次迭代 0.047ms / 500 次迭代 16ms。样例采用短采样并注明环境敏感。

## 同算子贯穿三层

本路线三层样例各自独立可跑；若要跟踪**同一算子**走完
kernel→op→framework，把各层样例的算子替换为同一个即可。
gelu_and_mul 的贯穿实证见 `tests/kernel_level/test_consistency.py`
与旗舰样例 [examples/b-fullstack](../b-fullstack/)。
