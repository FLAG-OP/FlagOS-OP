# P800 SDK 编译指南（SDK 就绪后使用）

## 前提

```bash
# 检查昆仑芯 SDK 是否可用
# [SDK-TODO] 替换为实际检测命令
xpu-sdk --version         # 或 kunlunxin-cc --version

# 确认编译器能产出 XPU 目标码
# [SDK-TODO] 替换为实际编译命令
xpu-cc --target=xpu80 -c xpu_kernel_template.cpp -o kernel.o
```

## 编译步骤

### 方式 A: 通过 torch.utils.cpp_extension（推荐）

```python
from torch.utils.cpp_extension import load

mod = load(
    name="xpu_kernels",
    sources=["xpu_kernel_template.cpp"],
    # [SDK-TODO] 替换为昆仑芯编译器标志
    extra_cuda_cflags=[  # 或 extra_xpu_cflags（SDK 可能新增）
        "-O3",
        "--target=xpu80",   # 替换为实际 XPU 架构
    ],
    build_directory="/tmp/xpu_build",
)
```

### 方式 B: 手动编译链接

```bash
# 1. 编译设备码
# [SDK-TODO] 替换 nvcc 为昆仑芯编译器
xpu-cc -O3 --target=xpu80 -c xpu_kernel_template.cpp -o kernel.o

# 2. 链接 torch extension
g++ -shared -o xpu_kernels.so kernel.o \
    -I$(python3 -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
    -L$(python3 -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])") \
    -ltorch -lc10

# 3. 测试
python3 -c "import xpu_kernels; print(dir(xpu_kernels))"
```

## SDK 到位后的检查清单

- [ ] 替换 `#include` 为昆仑芯 SDK 头文件
- [ ] 替换 `__global__` 为 XPU 设备函数语法（或确认 XMLIR 能翻译）
- [ ] 替换 kernel 启动 API（`<<<>>>` 或自定义 launch 函数）
- [ ] 替换编译器标志（`--target=xpu80` 等）
- [ ] 运行 `python3 examples/hw-kernel-example/example.py` 验证
- [ ] 更新 `configs/devices/p800-kunlunxin.yaml` 的 vendor_kernels 清单

## 参考: 其他厂商 SDK 的对应关系

| 厂商 | 编译器 | 设备函数标记 | 启动 API |
---|---|---|---|
| NVIDIA | nvcc | `__global__` | `<<<grid, block>>>` |
| 华为昇腾 | ascend-c | `__aicore__` | `aclrtLaunchKernel()` |
| 昆仑芯 | [SDK-TODO] | [SDK-TODO] | [SDK-TODO] |
