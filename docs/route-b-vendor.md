# 路线 B: 厂商语言 → vendor backend

## 适用场景

用厂商 SDK / C++ / 芯片专用语言写的 kernel（P800 上即昆仑芯
XPU C++ kernel，经 `xtorch_ops` / `.so` 暴露）。

## vendor backend 三要素

```python
class MyVendorBackend(Backend):
    @property
    def name(self):   return "myvendor"
    @property
    def vendor(self): return "myvendor"   # vendor 路线必须非 None
    def is_available(self):
        try: import my_vendor_lib; return True
        except ImportError: return False
    def my_op(self, obj, *args): ...      # obj 是调用层对象
```

注册: `OpImpl(op_name=..., impl_id="vendor.myvendor",
kind=BackendImplKind.VENDOR, vendor="myvendor",
priority=BackendPriority.VENDOR)`（100）

## 真实厂商 kernel 接入（以 P800 为例）

1. `routes/b_vendor/csrc/` 写 C++ kernel（pybind11 模板）
2. 厂商工具链编译产出 `.so`（参考 BUILD.md；内置算子库见 `/env/output/`）
3. backend 方法中调用（真实案例: `xtorch_ops.swiglu(x, out)`）
4. 算子层验证注册/选择/计数/语义

## 参考实现与样例

- 正式实现: `routes/b_vendor/`（audit 计数+委托 backend）
- 自包含样例: `examples/b-op/`
- 框架级样例: `examples/b-framework/`

## audit 模式（模板库的核心工具）

audit vendor = 计数 + 委托，双用途:

1. 验证 vendor 路线的注册/选择/调用链路
2. 框架层测试中拦截真实 vLLM 前向流量

委托目标由设备 profile 的 `vendor_delegate` 字段驱动
（p800 → `xtorch_ops.swiglu`；无厂商库芯片 → 自动退化 reference.torch）。

## 注意事项

1. `with_preference("vendor") + with_allowed_vendors("myvendor")`
   是精确选择特定 vendor 的正规方式
2. ⚠️ 本机 `xtorch_ops.swiglu` 独立调用**不写输出**（known-issues #5），
   框架级恒等断言必须用 reference 委托
