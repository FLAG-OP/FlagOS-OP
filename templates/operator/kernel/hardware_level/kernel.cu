// 硬件级模板 · CUDA C++ 设备码骨架（占位符 + 文字说明）
//
// 定位: NVIDIA 目标的类 C 设备码起点。
// ⚠️ P800/XPU: nvcc 可编译、无法执行（NVIDIA PTX 不被 XPU 识别，
//    known-issues #12）；P800 真设备码需昆仑芯 XTC SDK
//    （见 examples/hw-kernel-example/sdk_template/）。
//
// 编译: nvcc -O3 -shared -Xcompiler -fPIC kernel.cu -o libmyop.so
#include <cuda_runtime.h>

// [TODO] 设备端 kernel —— 与 reference.py 同语义。要点:
//   · 网格/块划分: 本库样例用 256 threads/block、grid=ceil(n/256)
//   · 访存: coalesced（相邻线程读相邻地址），必要时向量化
//   · 精度: 内部 fp32 累加，写出时 cast 回输入 dtype（与参考一致）
//   · 尾块: 边界判断放设备码内时，务必实测非整倍数 N（#15a 同类陷阱）
__global__ void my_op_kernel(/* [TODO] 输入/输出指针 + 维度参数 */) {
    // [TODO] 设备码
}

// [TODO] host 启动器 —— 要点:
//   · 绑定 stream（异步语义与 PyTorch 一致）
//   · 错误检查（cudaGetLastError / cudaPeekAtLastError）
extern "C" void my_op_launch(/* [TODO] ptr, n, cudaStream_t stream */) {
    // [TODO] my_op_kernel<<<grid, block, smem, stream>>>(...);
}
