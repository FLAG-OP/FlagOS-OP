# 路线 B: 厂商算子注册

[← 返回文档中心](index.md)

## 适用场景

用厂商 SDK / C++ / 芯片专用语言写的 kernel（经厂商 Python 绑定
或 `.so` 暴露；参考实例 P800 上为昆仑芯 XPU 栈的 `xtorch_ops`）。

这条路线上的 kernel 通常有两种来路：一种是 torch 级，用 C++ 调
ATen 组合出融合算子（如 b-fullstack）；另一种是硬件级，厂商用专用
语言写好、经 Python 绑定暴露。路线本身只负责注册与选择。

<a id="kernelspec"></a>
## 泛化接口 KernelSpec

`common/kernel_spec.py` 定义厂商 kernel 直测的声明式描述
（name / op 语义锚点 / out_mode / adapter / expected_ok / build），
从[设备 profile](device-profiles.md#fields) 的 `vendor_kernels` 段加载。
配套 `sentinel_check()` 见[测试体系](testing.md#sentinel)。

## 厂商算子注册 三要素

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

<a id="csrc"></a>
## 真实厂商 kernel 接入（以参考实例 P800 为例）

含 JIT 编译闭环: `tests/kernel_level/test_b.py` 经
`torch.utils.cpp_extension.load()` 编译 csrc → 加载 → 直测，
详见 [kernel 层文档](testing.md#kernel-level)。

1. `routes/b_vendor/csrc/` 写 C++ kernel（pybind11 模板）
2. 厂商工具链编译产出 `.so`（参考 BUILD.md；内置算子库见 `/env/output/`）
3. backend 方法中调用（真实案例: `xtorch_ops.swiglu(x, out)`）
4. 算子层验证注册/选择/计数/语义

## 参考实现与样例

- 正式实现: `routes/b_vendor/`（audit 计数+委托 backend）
- 自包含样例: `examples/b-op/`
- 框架级样例: `examples/b-framework/`
- kernel 层样例: [b-kernel](../examples/b-kernel/)（直测+哨兵+JIT 闭环）
- 全链路样例: [b-fullstack](../examples/b-fullstack/)（C++ 贯穿三层，附开发报告）
- 硬件级开发: [hw-kernel-example](../examples/hw-kernel-example/)（厂商原语 + SDK 模板）

## audit 模式（模板库的核心工具）

audit 是一个"什么都不算、但什么都记录"的厂商实现: 数值全部委托给
参考实现，唯一的动作是给每次调用计数。它存在的理由是分离两个问题
——"链路通不通"和"数值对不对"。先挂 audit 确认注册、选择、调用
整条链路是通的（此时输出必然正确，因为算的是参考实现），再换上
真实 kernel 时，任何输出变化都能干净地归因于 kernel 本身。

1. 验证 vendor 路线的注册/选择/调用链路
2. framework 验证（应用层）测试中拦截真实 vLLM 前向流量

委托目标由设备 profile 的 `vendor_delegate` 字段驱动
（未声明或厂商库缺失时自动退化 reference.torch）。

## 注意事项

1. `with_preference("vendor") + with_allowed_vendors("myvendor")`
   是精确选择特定 vendor 的正规方式
2. ⚠️ 本机 `xtorch_ops.swiglu` 独立调用**不写输出**（[known-issues #2](known-issues.md)），
   框架级恒等断言采用 reference 委托（厂商 kernel 无法保证数值恒等）

> ⚠️ 写 Triton kernel 时: 启动必须包 `torch_device_fn.device(...)` 上下文，
> 否则首次后的启动静默 no-op（[known-issues #11](known-issues.md)）。

---

**下一步**: [全链路指南](fullstack-guide.md)——三条路线的完整开发流程。 · [集成指南](integration.md)——开发完成后怎么交付
