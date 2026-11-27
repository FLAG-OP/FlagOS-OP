# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 自带反向） |
| 反向 | `native_layer_norm_backward` 原生委托（真机可用; 该栈未暴露 `layer_norm_backward` Python 绑定, 层语义同式） |
| forward | 借道 `aten::native_layer_norm` 三元组（mean/rstd 供反向; 与公开 API 语义一致） |
| 边界 | inference_mode 零命中（#19.3） |
