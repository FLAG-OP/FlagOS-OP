# 样例: B × kernel 层（核心新格）

厂商硬件语言 kernel 直测，四部分:

1. **JIT 编译闭环**: `routes/b_vendor/csrc/vendor_kernel.cpp` 经
   `torch.utils.cpp_extension.load()` 编译、加载、直调
2. **现成厂商 kernel 直测**: profile `vendor_kernels` 声明的清单
   （`xtorch_ops.*`），精度 vs PyTorch 语义参考
3. **哨兵健全性检查**: out_param 模式预填哨兵验真写入；return 模式
   验确定性与输入敏感——通用化检测 swiglu 型"不写输出" bug
4. **性能短采样**

本机实测: 3 个已知坏厂商 kernel（swiglu / silu / gelu_tanh_and_mul@XPU2）
全部被哨兵正确抓出；自研 C++ JIT 产物精度+哨兵全过。

```bash
python3 examples/b-kernel/example.py
```
