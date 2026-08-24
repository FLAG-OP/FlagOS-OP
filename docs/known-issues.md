# 本机已知问题清单（全部实测）

## 数值正确性类

### 1. 不要设 VLLM_FL_PREFER=flagos（P800）
FlagGems 的 rms_norm Triton dispatch 实现在真实 vLLM 前向中会失败。
容器默认值（非法值 `flagos|vendor`）恰好让所有算子回落
vendor.kunlunxin——是生产可用路径。测试自定义算子只用
`VLLM_FL_PER_OP` 精确钉住。

### 2. ⚠️ xtorch_ops.swiglu 不写输出（严重）
独立 eager 调用下完全不写输出张量（哨兵测试: torch.full 预填后
调用，全部元素保持原值；返回 int 0）。vendor.kunlunxin 的
silu_and_mul 在此栈上不可靠——生产 vLLM 表面正常疑似依赖 allocator
复用旧激活内存。建议向芯片厂商反馈。

### 3. embedding 静默越界
随机模型 tokenizer 词表(151669) > embedding(128256)，文本输入产生
越界 token id。vLLM/XPU 路径的 embedding 查找不做边界检查
（静默读越界内存）；CPU transformers 正确报 IndexError。
模板已统一 tokenize+词表钳制输入。

## 稳定性类

### 4. 跨进程后期 token 偶发非确定性
历史观察到 16+ token 偶发漂移；受控漂移实验（reference 8 连跑 +
vendor 4 连跑）全部 100% 确定——属低概率条件相关事件。
防御: 恒等断言前缀自适应 + 黄金多数票共识。

### 5. P800 单卡路径通道异常
TP=1 曾触发厂商 reshape_and_cache 通道级异常（status 719），
profile 固定 TP=2。

## 接口/文档类

### 6. 插件入口函数名不一致
dispatch README 文档写 `register_builtins`，代码实际查找
`register` / `vllm_fl_register`——外挂插件必须用后者。

### 7. vLLM v1 前向在子进程
主进程 torch.library 注册不传播到 EngineCore/Worker 子进程。
A1 框架级需 sitecustomize 注入桥。

### 8. 官方 kunlunxin backend 的 swiglu 调用存疑
即使按内置 backend 的调用方式，swiglu 输出语义与标准
silu(x1)*x2 不符（是问题 2 的衍生表现）。

### 9. 性能基准的分配器陷阱
逐次新分配大输出的长循环基准（如 500 次 × 128MB）会触发分配器
池增长，均值被抬高一两个数量级（实测同 kernel 0.047ms vs 16ms）。
基准一律用短采样（≤100 次）。
