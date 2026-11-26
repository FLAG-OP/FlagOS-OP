# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 自带反向） |
| mean/rstd 形状 | PPU 原生返回 **keepdim 形状**（(16,1) vs CPU (16,)）且**恒 fp32**（半精度输入亦然）——值广播等价, 测试 reshape 对拍 |
| 反向 | `aten::native_layer_norm_backward`（真机可用, 2026-11-25 卡6） |
| 配对 | 与 ops/layer_norm 一一对应（公开 API 消费原语, op 层第 1b 步取证） |
