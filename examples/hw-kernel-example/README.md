# hw-kernel-example: P800 硬件级算子开发

[← 样例索引](../README.md)

## 定位

P800 硬件级开发样例。本容器无昆仑芯 SDK，用 **xtorch_ops 厂商原语**
（预编译 XPU kernel）组合自定义算子，是当前环境下最接近硬件级的方式。

附 CUDA C++ 参考设备码（供 NVIDIA 环境直接复用）。

## 环境限制

| 开发方式 | 状态 | 说明 |
|---|---|---|
| 昆仑芯 SDK / XPU C++ | ❌ 本容器无 SDK | 真正的 P800 硬件级开发 |
| xtorch_ops 厂商原语 | ✅ 307 个可用 | **本样例使用**（预编译 XPU kernel 组合） |
| CUDA C++ (nvcc) | ⚠️ 编译✅ 执行❌ | XPU 无法运行 NVIDIA PTX |
| Triton 级 | ✅ 已验证 | → XMLIR → XPU 指令 |

## 运行

```bash
python3 examples/hw-kernel-example/example.py
```

## 内容

1. 用 xtorch_ops 厂商原语组合 fused_silu_and_mul
2. 精度 vs PyTorch 参考 + 哨兵检查
3. 性能对比（厂商原语 vs Triton vs PyTorch）
4. CUDA C++ 参考代码（附 NV 版设备码）
