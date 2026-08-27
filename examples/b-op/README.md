# 样例: B 路线 × 算子层

## 目标

定义自定义 **厂商算子注册**（芯片商语言 kernel 的 Python 接入层），
注册进 FlagOS 融合算子 并验证选择/计数/精度。

## 厂商算子注册 三要素

1. `vendor` 属性非 None（vendor 路线的身份标识）
2. `is_available()` 检测厂商硬件/库（生产中 import 厂商 lib）
3. 算子方法——生产中调用厂商 C++ kernel（如 `xtorch_ops.swiglu`）

## 运行

```bash
python3 examples/b-op/example.py
```

## 预期输出

```
注册: [..., 'vendor.myvendor']
选择: vendor.myvendor（with_allowed_vendors 精确钉住）
精度: max_err=0.0e+00
计数: {'silu_and_mul': 1}
```

## 关键点

1. `with_preference("vendor") + with_allowed_vendors("myvendor")`
   是精确选择特定 vendor 的正规方式
2. 真实厂商 kernel 接入见 `routes/b_vendor/csrc/`（C++ 模板 + 编译说明）
3. 本机已知: `xtorch_ops.swiglu` 独立调用不写输出（[已知问题 #2](../../docs/known-issues.md)），
   故样例用 fp32 语义实现替代
