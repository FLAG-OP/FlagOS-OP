# 已知问题清单

[← 返回文档中心](index.md)

## 关于本文件

本文件记录**参考实例（P800 / Kunlunxin 栈）**的实测问题，用作:

1. 该设备栈使用者的排障参考
2. 其他芯片接入时的检查清单模板——每接入一种新芯片，建议用同样的
   方法（哨兵检查 / drift_study / 黄金比对）建立该设备自己的记录

问题分类: 数值正确性 / 稳定性 / 接口与文档。

## 数值正确性类

### 1. VLLM_FL_PREFER 设为 flagos 会触发 rms_norm 失败（该栈）
FlagGems 的 rms_norm Triton dispatch 实现在真实 vLLM 前向中会失败。
容器默认值（非法值 `flagos|vendor`）恰好让所有算子回落
厂商算子注册——是生产可用路径。测试自定义算子可用
`VLLM_FL_PER_OP` 精确钉住目标算子（避免全局改动）。

### 2. 厂商 kernel 不写输出: xtorch_ops.swiglu（严重）
独立 eager 调用下完全不写输出张量（哨兵测试: torch.full 预填后
调用，全部元素保持原值；返回 int 0）。厂商算子注册 的
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
`register` / `vllm_fl_register`——外挂插件入口函数名需与代码查找一致。

### 9. vLLM v1 前向在子进程
主进程 torch.library 注册不传播到 EngineCore/Worker 子进程。
aten 路线框架级需 sitecustomize 注入桥（本库已内置）。

### 10. 性能基准的分配器陷阱
逐次新分配大输出的长循环基准（如 500 次 × 128MB）触发分配器
池增长，均值被抬高一两个数量级（实测同 kernel 0.047ms vs 16ms）。
两层防护: ① 短采样（≤100 次）；② 多用例不共进程——
[perf_run](performance-regression.md) 对每个用例起独立子进程。
短采样单独不够: 同一 C++ kernel 单独跑 0.311ms，与其他用例同进程
顺序混跑（仍 20+100 短采样）被抬到 1.688ms；A2 参考实现
3.69ms → 317ms。分配器状态污染跨用例传播，隔离后恢复健康态。

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

该缺陷已进入 [intake 负例](ai-intake.md): AI 生成 kernel 若漏包此
上下文，[intake 验证](ai-intake.md)的哨兵检查会在 kernel 层直接
拦截（确定性 ✗），返回机器可读的 BLOCKED 原因供生成侧迭代。

### 16. A2 `call_op` 长循环偶发挂起
测量分发开销时发现: 对同一 op 连续 `call_op` 约 300 次的循环在
本栈多次挂起（Ctrl-C 也难中断，需 kill 进程）；同一调用 100 次
以内稳定。挂起出现在 `with_preference` 上下文内的纯 Python 循环，
不涉及新 kernel 编译，复现条件尚不稳定（有时 300 次也能跑完）。
缓解: `scripts/bench_dispatch.py` 默认 100 次短循环；如需复现
排查，从 n=200 起逐步加长。

<a id="method"></a>
### 12. CUDA C++ 设备码（NVIDIA）无法在 P800/XPU 执行
nvcc 编译 NVIDIA CUDA C++ 通过，但 P800 的 XPU 硬件无法识别
NVIDIA PTX 指令集（报 `invalid device function`）。XMLIR 兼容层
只翻译 ATen/Triton 中间表示，不翻译 nvcc 二进制。

P800 真正的硬件级开发需昆仑芯 SDK（本容器未提供）。当前最接近
硬件级的方式是用 xtorch_ops 厂商原语组合（见 hw-kernel-example）
或 Triton 级开发（→ XMLIR → XPU 指令）。

### 13. FlagGems `gelu(approximate="tanh")` 在本栈链接失败
FlagGems 4.2.1rc0 的 `gelu_tanh` kernel 使用 `tanh` 内建函数，而本栈
xtriton 的 XPU libdevice 表中该 fp32 重载映射到占位符符号
`"Unsupported"`，`xpu*-elfconv` 链接时报
`ld.lld: error: undefined symbol: Unsupported`。同栈实测:
`gelu(none)` / `silu` / `softmax` / `bmm` 均正常，仅 tanh 路径损坏。

影响: A1（gelu_tanh）与 A2（gelu_and_mul）**没有同语义的 FlagGems
生产基线**；自研 kernel 用 `1 - 2/(exp(2x)+1)` 手写 tanh 故不受影响。

### 14. XPU Triton 编译错误被 `NameError: sys` 掩盖
两因叠加: ① xtriton `backends/xpu/compiler.py` 的 `run_cmd()` 使用了
未 import 的 `sys`；② torch_xmlir 符号改写会重新导入 xpu backend，
产生一份**不在 `sys.modules`** 的 compiler 实例——对
`triton.backends.xpu.compiler` 打补丁无效。结果: 任何 elfconv 报错
（含 #13）都被 NameError 吞掉。

修复: `common/xpu_compat.py` 经 `triton.backends` 注册表直达第二实例
注入 `sys`（`common/device.py` 导入时自动生效），此后真实错误可见。

### 15. ⚠️ XPU 栈两个 Triton 正确性陷阱（softmax 实测发现）

**a) 尾块 masked load + `tl.sum` 归约错误**: N 非 BLOCK_N 整数倍时，
尾块中 masked load 的结果污染归约——`other=0.0`、`other=-1e30`、
`tl.where` 显式清洗**三种防护全部无效**；无 load 的纯归约正常，
`tl.max` 归约正常，仅 `tl.sum` 中招。实测 N=3072/BN=2048 输出 inf，
N=5120 误差 0.6。原 softmax 样例的测试形状恰好全是整倍数，漏测尾块。

**b) autotuner 选出非法 `num_warps=5`**: 配置表只定义了 4/8/16，
`best_config` 却显示 `num_warps: 5`（非 2 的幂），导致错误执行且
**每次调优结果不同**——同一 kernel 时对时错，极具迷惑性。

修复（softmax-fullstack 已落地）: wrapper 把输入 pad 到 2048 整数倍
（实体填充 -60000，exp 后贡献为 0），kernel 固定 `BLOCK_N=2048`——
全程无 mask、无调优，确定性路径；整数倍 N 零开销。测试形状补充
3072/5000/5120 非整倍数回归。

## 通用检测方法

| 问题类型 | 检测工具 |
|---|---|
| kernel 不写输出 / 非确定 | kernel 层哨兵检查（`sentinel_check`） |
| 按设备分化的正确性 | 逐物理设备跑 `--level kernel` |
| 栈升级数值漂移 | 黄金回归（`build_golden` + 框架测试自动比对） |
| 跨进程非确定性 | `scripts/drift_study.py` |
| 裸 Triton 启动静默 no-op | 哨兵检查（zeros 预填看零占比）+ 对照 `flag_gems` 同算子 |
| embedding 越界 | tokenize 后比对词表上限；CPU transformers 交叉验证 |
| FlagGems 某算子链接失败 | `common/xpu_compat` 补丁后看 elfconv 真实 stderr；对照 #13 |
| 尾块归约静默错误 | 精度探针含非整倍数 N（`accuracy_report.py`）；对照 #15 |
| call_op 长循环挂起 | 短循环（≤100 次）规避；排查见 #16 |

---

**下一步**: [样例索引](../examples/README.md)（可运行范本）·
[AI 生成算子接入](ai-intake.md)（缺陷自动拦截）。
