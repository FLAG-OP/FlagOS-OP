# 硬件级: 本算子未实现（置空说明）

`contiguous` 是 layout 重排/拷贝。MLU 上由 runtime 的设备码承担，
本机无独立厂商原语，故硬件级置空。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py) |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py) |
| 硬件级 | ⬜ 置空 | 本目录 |
