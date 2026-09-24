# 硬件级: 本算子未实现（置空说明）

`type_as` 是纯 elementwise cast。MLU 上该语义由框架/runtime 的
`to`/`copy_` 设备码承担，本机 `torch_mlu` 未暴露同语义的独立厂商原语
（`torch_mlu_ops` 中无 `type_as`/cast 原语），故硬件级置空。

如需接入: 按[样板骨架](../../../../templates/operator/kernel/hardware_level/BUILD.md)
操作。当前本算子三级状态:

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py)（ATen `.to`） |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py)（自研，含 #11） |
| 硬件级 | ⬜ 置空 | 本目录（无厂商原语，见上） |
