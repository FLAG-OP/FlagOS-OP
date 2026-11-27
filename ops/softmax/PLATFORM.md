# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 解析反向） |
| 重载集 | 本栈 `int/Dimname/int_out`（**无 int_dtype**）; `Library.impl` 必须按 **`softmax.int`** 重载名绑定——裸名绑定注册到无 schema 空壳, 调用永不命中（op 层实证） |
| dtype 语义 | h2f 仅 Half（厂商限制, 实证）; bf16→fp32 走同 dtype 计算 + cast |
| 反向 | dX = Y⊙(g − (Y⊙g).sum(dim))（输出保存复用, 免重算） |
