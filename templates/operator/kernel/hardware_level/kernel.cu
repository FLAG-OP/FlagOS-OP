// 硬件级实现 · CUDA C++（类 C 设备码参考实现）
//
// ⚠️ 目标定位: NVIDIA GPU。在 P800/XPU 上 nvcc 可编译但无法执行
// （NVIDIA PTX 指令集不被 XPU 识别，known-issues #12）。
// P800 当前可运行的"最接近硬件级"路径见同目录 xtorch_binding.cpp。
//
// 编译: nvcc -O3 -shared -Xcompiler -fPIC kernel.cu -o libmyop.so
#include <cuda_runtime.h>
#include <cmath>

__global__ void my_op_kernel(const float* __restrict__ x,
                             const float* __restrict__ g,
                             float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xf = x[i];
    // <TODO: 与 reference.py 同语义的 fp32 计算>
    y[i] = xf * g[i];
}

extern "C" void my_op_launch(const float* x, const float* g, float* y,
                             long n, cudaStream_t stream) {
    long blocks = (n + 255) / 256;
    my_op_kernel<<<blocks, 256, 0, stream>>>(x, g, y, n);
}
