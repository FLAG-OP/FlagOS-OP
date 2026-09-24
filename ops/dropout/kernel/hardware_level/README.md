# 硬件级: 本算子未实现（置空说明）

dropout 含 RNG，本机无同语义厂商原语（`torch_mlu_ops` 无 dropout），
故硬件级置空。RNG 由 Triton Philox 承担。

| 级别 | 状态 | 位置 |
|---|---|---|
| torch 级 | ✅ | [../torch_level.py](../torch_level.py) |
| Triton 级 | ✅ | [../triton_level.py](../triton_level.py)（Philox） |
| 硬件级 | ⬜ 置空 | 本目录 |
