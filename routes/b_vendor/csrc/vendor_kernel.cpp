// 厂商 C++ kernel 模板（设备无关占位; 真实编译替换为厂商 SDK 头与 API）
#include <torch/extension.h>
#include <ATen/ATen.h>

torch::Tensor gelu_and_mul_vendor(torch::Tensor x, torch::Tensor gate) {
    auto xf = x.to(torch::kFloat32);
    auto inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf);
    auto gelu = 0.5 * xf * (1.0 + at::tanh(inner));
    return (gelu * gate.to(torch::kFloat32)).to(x.scalar_type());
}

// vLLM SiluAndMul 语义: x[..., :d] 过 silu 后乘 x[..., d:]
torch::Tensor silu_and_mul_vendor(torch::Tensor x) {
    auto sizes = x.sizes().vec();
    int64_t d = sizes.back() / 2;
    sizes.pop_back();
    sizes.push_back(d);
    auto xf = x.to(torch::kFloat32);
    auto x1 = xf.narrow(-1, 0, d);
    auto x2 = xf.narrow(-1, d, d);
    auto silu = x1 * at::sigmoid(x1);
    return (silu * x2).to(x.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_and_mul", &gelu_and_mul_vendor, "gelu_and_mul vendor kernel");
    m.def("silu_and_mul", &silu_and_mul_vendor, "silu_and_mul vendor kernel");
}
