# 路线 B: 芯片商语言算子 → vendor backend 接入

## 适用场景

用厂商 SDK / C++ / 芯片专用语言写的 kernel。

## vendor backend 三要素

```python
class MyVendorBackend(Backend):
    @property
    def name(self):   return "myvendor"
    @property
    def vendor(self): return "myvendor"   # vendor 路线必须非 None
    def is_available(self): ...           # 检测厂商库/硬件
    def my_op(self, obj, *args): ...      # obj 是调用层对象

OpImpl(op_name="my_op", impl_id="vendor.myvendor",
       kind=BackendImplKind.VENDOR, vendor="myvendor",
       priority=BackendPriority.VENDOR)   # 100
```

## 组成

| 文件 | 作用 |
|---|---|
| `csrc/` | 厂商 C++ kernel 模板 + pybind11 + 编译说明 |
| `backend/audit_vendor.py` | 可运行演示: 计数+委托（委托目标来自设备 profile） |
| `backend/register_ops.py` | vendor 注册模板 |

audit 的 silu_and_mul 委托由设备 profile 的 `vendor_delegate` 字段驱动
（p800 → `xtorch_ops.swiglu`；无厂商库芯片 → reference.torch）。

## 测试入口

```bash
python3 run.py --route b --level op --device p800-kunlunxin
python3 run.py --route b --level framework --device p800-kunlunxin
```
