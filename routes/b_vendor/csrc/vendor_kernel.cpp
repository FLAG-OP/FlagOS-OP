// 厂商 C++ kernel 模板（设备无关占位; 真实编译替换为厂商 SDK 头与 API）
#include <torch/extension.h>
#include <ATen/ATen.h>

torch::Tensor gelu_and_mul_vendor(torch::Tensor x, torch::Tensor gate) {
    auto xf = x.to(torch::kFloat32);
    auto inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf);
    auto gelu = 0.5 * xf * (1.0 + at::tanh(inner));
    return (gelu * gate.to(torch::kFloat32)).to(x.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_and_mul", &gelu_and_mul_vendor, "gelu_and_mul vendor kernel");
}
