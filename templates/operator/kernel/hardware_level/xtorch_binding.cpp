// 硬件级模板 · 厂商 C++ 绑定骨架（占位符 + 文字说明）
//
// 定位: P800/XPU 当前可运行的"最接近硬件级"路径——host 侧 C++ 调用
// 厂商预编译算子库（xtorch_ops / infer_ops）；真厂商语言设备码待 XTC。
//
// 查 API: /env/output/infer_ops/include/infer_ops*.h
//   （同语义 kernel 如 infer_swiglu_sdnn；include 三链与编译命令
//    见同目录 BUILD.md，本库已验证 #include <infer_ops.h> 可编译）
//
// ⚠️ 厂商 kernel 两个实测坑:
//   · out 参数模式可能"不写输出"——上哨兵检查（known-issues #2）
//   · 按物理设备分化——多卡都要跑（known-issues #5）
#include <torch/extension.h>

torch::Tensor my_op_vendor(torch::Tensor x, torch::Tensor g) {
    // [TODO] 调用厂商同语义 kernel:
    //   1. 确认签名与 out 模式（return / out 参数）
    //   2. 构造厂商 context / stream
    //   3. 调用并处理错误
    TORCH_CHECK(false, "my_op_vendor: 模板占位，未实现");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_op", &my_op_vendor, "my_op (vendor binding skeleton)");
}
