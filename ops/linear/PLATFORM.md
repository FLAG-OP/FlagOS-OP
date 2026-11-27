# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 组合反向） |
| redispatch 键 | **后端键 CUDA**（CompositeExplicitAutograd 落组合分解, 累加序不同 → 半精度分化, bf16 LN 差 0.031 同源实证） |
| 反向 | 组合式（gW / Wᵀg / bias 求和; aten::matmul 组合, 无同键递归） |
| 边界 | inference_mode 零命中（#19.3）; 3D batched no-bias bf16 err 1.5e-5 |
