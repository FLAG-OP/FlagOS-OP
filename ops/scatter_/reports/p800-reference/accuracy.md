# 精度

scatter 不做算术，黄金比较使用 rtol=0、atol=0；整数不转 float。
CPU oracle 不能证明设备内核正确。P800 Triton 黄金回放已通过 41/41。
CPU 黄金已生成 41 个，SHA256 校验及原生回放 41/41 通过
（torch 2.15.0+cpu / Python 3.12）。
重复 tensor-src 索引不比较某个特定胜者；scalar-src 重复索引可精确比较。
设备额外 11 项边界验证通过，重复 tensor-src 验证赢家属于输入值集合。
