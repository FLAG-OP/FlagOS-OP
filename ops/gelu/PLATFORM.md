# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 自带解析反向; inference_mode 零命中 = #19.3 边界） |
| Triton tanh | PPU triton 3.6.0 `tl.math` **无 tanh**（erf 有）→ 手写稳定 tanh `1-2/(exp(2\|x\|)+1)`（#13 同款, IEEE inf 算术溢出安全） |
| schema | `approximate` 为 **keyword-only**（位置传参在 C++ boxed redispatch 层同样报错） |
| 非有限值 | `gelu(+inf, none)`: 官方 CPU=NaN（golden 锚）vs 设备 erf 公式=+Inf——不定式两种合法 IEEE 解释, INFO case 不硬判（gen_golden 注） |
| FlagFix #5867 | 修复对象=ascend 后端 CANN pow 查表（geglu/gelu_and_mul/gelu.py）; **平台边界=昇腾**; PPU 不消费 CANN libdevice, 不可复现; 迁移的是写法规约（fp32 显式常量/无 int 指数）; 真武验收按本算子三层+黄金独立完成 |
| 性能 | `1.08–1.10×` 仅限 8192×8192 半精度（复跑 fp16=1.082×, 历史 JSON bf16=1.098×）; 小形状/4096×4096/fp32 走 native |
