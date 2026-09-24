# 硬件级: 本算子未实现（置空说明）

`copy_` 是纯数据搬运。MLU 上由 runtime 的 copy 设备码承担，本机无独立
厂商原语，故硬件级置空。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py)（原生 copy_） |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py)（含 2D swiz） |
| 硬件级 | ⬜ 置空 | 本目录 |
