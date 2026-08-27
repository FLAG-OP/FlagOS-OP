# hw-kernel-example: 手写硬件语言设备码

[← 样例索引](../README.md)

## 定位

全库唯一的**硬件级**开发样例——手写 CUDA C++ `__global__` 设备函数，
覆盖 vector_add 和 fused_silu_and_mul 两个 kernel。

## 实测结论（P800/XPU 栈）

| 阶段 | 结果 | 说明 |
|---|---|---|
| JIT 编译 | ✅ | nvcc 编译通过，产出 .so |
| 设备执行 | ❌ | XPU 无法运行 NVIDIA PTX |

**原因**: XMLIR 兼容层只翻译 ATen/Triton 中间表示，不翻译 nvcc
直接产出的 NVIDIA 二进制码。

**可行路径**:
1. 厂商 SDK: 使用昆仑芯自家编译器（非 nvcc）
2. Triton 级: 当前环境下最接近硬件的开发方式（→ XMLIR → XPU）
3. torch 级: C++ 调 ATen（如 [b-fullstack](../b-fullstack/)）

## 运行

```bash
python3 examples/hw-kernel-example/example.py
```

## CUDA C++ 参考实现

本样例包含完整的 `__global__` 设备函数（vector_add + fused_silu_and_mul），
在有 NVIDIA GPU 或厂商 SDK 的环境中可直接编译执行。
