# 测试范围与结果

P800 gems5：kernel 41/41、边界 11/11、A1 41/41、mini consumer 1/1。
覆盖浮点、整数、bool、负 dim、空/标量、非连续、partial index、尾块、
超出 fp32 精度的 int64 标量、越界拒绝、大整块、4D/5D。
原地返回对象、stride、storage offset、A1 版本递增均有断言。
尾块修复后 kernel/A1 连续 10 轮通过。

CPU：元数据 pytest 7 项；Triton 3.9.0 解释器 34/34 非空、非 bf16 用例。
CI 等价包化导入扫描 17/17 通过。
设备软件栈核对 `check_env --device p800-kunlunxin-gems5` 通过。

应用层为 token-selection mini consumer，基线/注册后精确比较且命中计数为 1。
此为 C1。另有真实 vLLM C2：随机权重 Qwen3，生成阶段 src 命中 8 次，
两请求共 16 个 token 与基线精确一致；见 vllm.md 与 vllm-c2.json。
训练/反向传播不是本次验证范围。
结构化结果见 `test-p800.json`；运行方式见上级 README。
