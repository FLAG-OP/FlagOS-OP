# 平台绑定清单（Platform Binding）— thead-ppu

> 环境锁定: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 用卡 6（物理卡号口径, sdpa PLATFORM §3c-2）;
> 跨平台移植时逐项重验, 不要把单平台实验结果当成普适行为。

| 绑定项 | 结论 |
|---|---|
| 注册 key | `AutogradCUDA`（thead 范式, 解析反向） |
| h2f 边界 | 厂商 h2f **仅支持 Half**（bf16/fp32 报 "conversion is supported for Half type only", 实证）→ 回退 fp32 域计算 + cast（数学等价） |
| 调用关系 | F.softmax → aten::softmax（ops/softmax）→ 内部落点 _softmax 原语; 单独注册原语时公开 API 不经 Python 注册面（op 层 public_hits=1 取证） |
