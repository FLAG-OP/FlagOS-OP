from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 多数 AI 加速卡伪装为 CUDA 设备（XPU/NPU 等），用 CUDAExtension +
# 厂商编译器包装；真实环境需指向厂商工具链（见 BUILD.md）。
setup(
    name="my_vendor_ops",
    ext_modules=[
        CUDAExtension(
            name="my_vendor_ops",
            sources=["vendor_kernel.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
