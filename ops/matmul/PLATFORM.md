# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 分形态组合反向） |
| 反向覆盖 | 2D@2D / ND@2D / 2D@ND(广播) / ND@ND 同批 / 批广播求和还原 / 1D reshape |
| Triton GEMM | 不自研——PPU `tl.dot` 精度降级（64³ rel≈7e-4, sdpa §3c）, GEMM 是厂商库强项 |
| 公共目录 | PR #35（P800）/ PR #53（MLU590）在途, 合入按 MERGE.md 七步; PPU 范围不互相推广 |
