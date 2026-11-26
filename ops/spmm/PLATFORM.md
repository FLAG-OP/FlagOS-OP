# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, COO 组合反向） |
| 原生缺口 | `cusparseSpMM_bufferSize not implemented`（fp16 稳定/fp32 冷启动不稳定）+ 组合探针 **C++ abort**（returncode -6）——原生不可作生产, 证据见 probes/thead_dense_p0_report.md §三 |
| 累加精度 | fp32 累加后 cast（与 CUDA 稀疏惯例一致; bf16 逐加序噪声实测 0.0625）; 判卷用 **fp64 独立参考 + 5e-2 rel**（乘加放大档） |
| CSR 边界 | 空行→零行 / 重复列索引→值累加（手工构造对拍）/ 无序索引→序无关; CSR `.t()` 返回 CSC（layout 翻转, 测试已对齐） |
| 宿主绑定 | torch.sparse.mm → `aten::_sparse_mm`; torch.spmm = torch.mm 别名（稠密行为保留, 不注册——op 层 dense_mm_untouched 断言） |
