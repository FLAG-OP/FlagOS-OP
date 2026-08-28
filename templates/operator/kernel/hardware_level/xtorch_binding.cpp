// 硬件级实现 · 厂商 C++ 绑定（P800/XPU 当前可运行路径）
//
// 形态: torch cpp_extension 加载本文件，host 侧调用厂商预编译算子库
// （xtorch_ops / infer_ops）。真厂商语言设备码需昆仑芯 SDK（XTC/xcc，
// 本容器未提供；模板见 examples/hw-kernel-example/sdk_template/）。
//
// 编译（torch.utils.cpp_extension.load，参照 routes/b_vendor/csrc）:
//   头文件三链（本库已验证可编译）:
//     -I /env/output/infer_ops/include
//     -I <torch_xmlir路径>/xhpc/xdnn/include
//     -I <torch_xmlir路径>/xre/include
//   链接: -L<torch_xmlir路径>/xre/so
#include <torch/extension.h>

torch::Tensor my_op_vendor(torch::Tensor x, torch::Tensor g) {
    // <TODO: 调用厂商算子库，如 xtorch_ops / infer_ops 的同语义 kernel>
    // 以下为占位的 ATen 回退（保证模板可编译运行，替换为厂商调用）
    auto xf = x.to(torch::kFloat32);
    auto out = xf * g.to(torch::kFloat32);
    return out.to(x.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_op", &my_op_vendor, "my_op (vendor binding template)");
}
