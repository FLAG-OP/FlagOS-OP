# 厂商 kernel 编译说明（以 Kunlunxin P800 为例）

## 真实厂商案例

内置 kunlunxin vendor 算子来自 `/env/output/`:

```
/env/output/infer_ops/libapiinfer.so      # 厂商推理算子库
/env/output/xft_blocks/libxft_blocks.so   # transformer block 算子
/env/output/xtorch_ops*.whl               # pybind 封装
```

调用方式:

```python
import xtorch_ops
out = torch.empty(..., dtype=x.dtype, device=x.device)
xtorch_ops.swiglu(x, out)
```

## 自研厂商 kernel 步骤

1. 用厂商 SDK 编写 C++ 源码（`vendor_kernel.cpp`）
2. 配置厂商编译工具链（include/lib 路径）
3. `python setup.py build_ext --inplace` 产出 `.so`
4. backend `is_available()` 里 `import my_vendor_ops`
5. 算子方法里调用 `my_vendor_ops.xxx(...)`

缺厂商编译头文件时 `setup.py` 无法编译属预期——先用纯 Python 的
audit backend 打通注册链路，再替换为编译好的 `.so` 调用。
