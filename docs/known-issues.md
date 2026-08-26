# 已知问题清单

[← 返回文档中心](index.md)

## 关于本文件

本文件记录**参考实例（P800 / Kunlunxin 栈）**的实测问题，用作:

1. 该设备栈使用者的排障参考
2. 其他芯片接入时的检查清单模板——每接入一种新芯片，建议用同样的
   方法（哨兵检查 / drift_study / 黄金比对）建立该设备自己的记录

问题分类: 数值正确性 / 稳定性 / 接口与文档。

## 数值正确性类

### 1. 不要设 VLLM_FL_PREFER=flagos（该栈）
FlagGems 的 rms_norm Triton dispatch 实现在真实 vLLM 前向中会失败。
容器默认值（非法值 `flagos|vendor`）恰好让所有算子回落
vendor backend——是生产可用路径。测试自定义算子只用
`VLLM_FL_PER_OP` 精确钉住。

### 2. 厂商 kernel 不写输出: xtorch_ops.swiglu（严重）
独立 eager 调用下完全不写输出张量（哨兵测试: torch.full 预填后
调用，全部元素保持原值；返回 int 0）。vendor backend 的
silu_and_mul 在此栈上不可靠——生产 vLLM 表面正常疑似依赖 allocator
复用旧激活内存。建议向芯片厂商反馈。

### 3. 同类: xtorch_ops.silu 不写输出
out 参数模式调用后输出全零未写入（sentinel 0/65536）。

### 4. embedding 静默越界
随机模型 tokenizer 词表(151669) > embedding(128256)，文本输入产生
越界 token id。该栈的 embedding 查找不做边界检查（静默读越界
内存）；CPU transformers 则正确报 IndexError。模板已统一
tokenize+词表钳制输入。

### 5. 厂商 kernel 按物理设备分化: gelu_tanh_and_mul
XPU1 上正常（vs 参考 err≈0.125、确定）；XPU2 上非确定且输出 inf。
kernel 层哨兵检查稳定抓出——**逐设备直测的必要性实证**。

## 稳定性类

### 6. 跨进程后期 token 偶发非确定性
历史观察到 16+ token 偶发漂移；受控漂移实验（reference 8 连跑 +
vendor 4 连跑）全部 100% 确定——属低概率条件相关事件。
防御: 逐 prompt 自适应断言前缀 + 黄金多数票共识。

### 7. 单卡路径通道异常
TP=1 曾触发厂商 reshape_and_cache 通道级异常，设备 profile 固定 TP=2。

## 接口/文档类

### 8. 插件入口函数名不一致
dispatch 文档写 `register_builtins`，代码实际查找
`register` / `vllm_fl_register`——外挂插件必须用后者。

### 9. vLLM v1 前向在子进程
主进程 torch.library 注册不传播到 EngineCore/Worker 子进程。
aten 路线框架级需 sitecustomize 注入桥（本库已内置）。

### 10. 性能基准的分配器陷阱
逐次新分配大输出的长循环基准（如 500 次 × 128MB）触发分配器
池增长，均值被抬高一两个数量级（实测同 kernel 0.047ms vs 16ms）。
基准一律用短采样（≤100 次）。

### 11. ⚠️ 裸 Triton 启动缺 device 上下文 → 静默 no-op（严重）
裸 `@triton.jit` kernel 启动若不包
`flag_gems.runtime.torch_device_fn.device(...)` 上下文，**首次编译启动
正常，之后所有启动静默 no-op**（输出为未初始化内存，不报任何错）。
FlagGems 算子内部都包了上下文故从未暴露；自写 kernel 必踩。
表现极具迷惑性：首次结果正确、性能基准测到假数据（实测同一 kernel
假 0.19ms vs 真 0.030ms）。修复（bmm-fullstack 发现，见其 README）:

```python
from flag_gems.runtime import torch_device_fn
with torch_device_fn.device(x.device):
    my_kernel[grid](...)
```

<a id="method"></a>
## 通用检测方法（任何芯片栈适用）

| 问题类型 | 检测工具 |
|---|---|
| kernel 不写输出 / 非确定 | kernel 层哨兵检查（`sentinel_check`） |
| 按设备分化的正确性 | 逐物理设备跑 `--level kernel` |
| 栈升级数值漂移 | 黄金回归（`build_golden` + 框架测试自动比对） |
| 跨进程非确定性 | `scripts/drift_study.py` |
| 裸 Triton 启动静默 no-op | 哨兵检查（zeros 预填看零占比）+ 对照 `flag_gems` 同算子 |
| embedding 越界 | tokenize 后比对词表上限；CPU transformers 交叉验证 |
