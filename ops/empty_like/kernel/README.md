# kernel 级别说明: 本算子无 Triton / 硬件级实现

`empty_like` 是**分配/元数据**算子：返回未初始化内存，正确性只由
**元数据**（shape / dtype / stride / device / memory_format）定义，
没有可编译的元素级设备码。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [torch_level.py](torch_level.py)（meta 推 stride + empty_strided） |
| Triton 级 | ⬜ 不适用 | 无算术、无需要写出的元素 |
| 硬件级 | ⬜ 不适用 | 分配由 runtime 承担 |

> 与 FlagGems 的差异：FlagGems 为了满足「必须有 device code」会启动一个
> 写 0 的 dummy kernel；FlagOS-OP 不强制 device code，故此处不引入无意义
> kernel。参考实现见 [../reference.py](../reference.py)。
