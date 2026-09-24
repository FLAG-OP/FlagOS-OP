# 硬件级: 本算子未实现（置空说明）

`clone` 是纯数据搬运。MLU 上由 runtime 的 copy 设备码承担，本机
`torch_mlu` 未暴露独立的 clone 原语，故硬件级置空。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py)（empty_like + copy_） |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py) |
| 硬件级 | ⬜ 置空 | 本目录 |
