# 硬件级: 本样例未实现（置空说明）

softmax 在本机厂商算子库（xtorch_ops / infer_ops）中未找到同语义
原语，且昆仑芯 XTC SDK 未提供，故硬件级置空。

如需接入: 按[样板骨架](../../../../templates/operator/kernel/hardware_level/BUILD.md)
操作——CUDA C++ 目标可用 kernel.cu 骨架（P800 执行受限
[#12](../../../../docs/known-issues.md)）；厂商库绑定走
xtorch_binding.cpp 骨架（include 三链已验证）。

当前本算子的三级状态:

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py)（ATen F.softmax） |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py)（自研，尾块安全） |
| 硬件级 | ⬜ 置空 | 本目录（无厂商原语，见上） |
