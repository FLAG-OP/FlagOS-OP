# 黄金数据

在仓库根目录执行 `python3 ops/scatter_/script/gen_golden.py`。
CPU 原生 scatter_ 生成输入、输出和 SHA256 manifest；`data/` 不入库。
规格见 `inputs_spec.yaml`，实际构造入口为 `test/cases.py`。
浮点和整数均逐元素精确比较，不使用 fp32 转换（避免 int64 精度损失）。
