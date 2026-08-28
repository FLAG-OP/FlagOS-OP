# 硬件级编译指南

| 目标 | 文件 | 本机状态 |
|---|---|---|
| NVIDIA GPU | [kernel.cu](kernel.cu) | nvcc 编译 ✓ / P800 执行 ✗（[#12](../../../../docs/known-issues.md)） |
| P800/XPU（厂商库绑定） | [xtorch_binding.cpp](xtorch_binding.cpp) | ✅ 可编译可执行（ATen 回退占位） |
| P800 真设备码 | [sdk_template](../../../../examples/hw-kernel-example/sdk_template/BUILD.md) | 待昆仑芯 SDK（XTC/xcc） |

## xtorch_binding.cpp 编译要点

```python
from torch.utils.cpp_extension import load
mod = load(name="my_op_vendor",
           sources=["kernel/hardware_level/xtorch_binding.cpp"],
           extra_include_paths=[
               "/env/output/infer_ops/include",       # 厂商算子 C++ API
               "<torch_xmlir>/xhpc/xdnn/include",     # xpu/refactor 开发头
               "<torch_xmlir>/xre/include",           # xpu/runtime.h
           ],
           extra_ldflags=["-L<torch_xmlir>/xre/so"])
```

三条 include 链在本库已验证可通过 `#include <infer_ops.h>` 的独立编译。
