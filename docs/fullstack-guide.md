# 全链路开发指南: 一个算子从源码到真实推理

[← 返回文档中心](index.md)

> 可运行范本: [examples/b-fullstack](../examples/b-fullstack/)
> （`python3 examples/b-fullstack/example.py`，约 3-4 分钟）

<a id="pipeline"></a>
## 流水线总览

```
   你的 kernel（C++ / Triton / 厂商库）
        │  JIT 编译（cpp_extension.load）或 pointwise_dynamic
        ▼
┌─ L0 kernel 层 ─────────────────────────────┐
│  直测: 精度 vs PyTorch 参考 · 哨兵 · 性能   │  run.py --level kernel
└──────────────┬─────────────────────────────┘
               │  注册（aten impl / OpImpl / vendor backend）
               ▼
┌─ L2 op 层 ────────────────────────────────┐
│  注册 · 策略钉选（PER_OP）· fallback       │  run.py --level op
└──────────────┬─────────────────────────────┘
               │  注入（sitecustomize / PLUGIN_MODULES）
               ▼
┌─ L4 framework 层 ─────────────────────────┐
│  真实推理: 调用计数 · 输出比对 · 黄金回归   │  run.py --level framework
└───────────────────────────────────────────┘
               ＋  跨层一致性（--consistency）
```

## 分步指引（以厂商语言 C++ 为例）

### 第 1 步: 写 kernel 并直测（L0）

1. 参考 [routes/b_vendor/csrc/vendor_kernel.cpp](../routes/b_vendor/csrc/vendor_kernel.cpp)
   写 C++ kernel（pybind11 绑定）
2. 在[设备 profile](device-profiles.md) 的 `vendor_kernels` 段声明
   （name / op 语义锚点 / out_mode / adapter / build: csrc）
3. 跑 `run.py --route b --level kernel`：
   - **精度**: vs PyTorch 语义参考，多 shape×dtype
   - **[哨兵检查](testing.md#sentinel)**: 确定性 / 输入敏感 / out 真写入
     （能抓出"不写输出"类厂商 bug——参考 case 中 3 个坏 kernel 全被它检出）
   - 性能短采样（≤100 次，避免[分配器陷阱](known-issues.md)）

### 第 2 步: 注册进 dispatch（L2）

1. 写 vendor backend（继承 `Backend`，`vendor` 属性 + `is_available` + 算子方法），
   参考 [fullstack_plugin.py](../examples/b-fullstack/fullstack_plugin.py)
2. 注册 `OpImpl(kind=VENDOR, vendor="my-cpp", priority=100)`
3. 验证钉选: `with_preference("vendor") + with_allowed_vendors("my-cpp")`
   或 `VLLM_FL_PER_OP="op=vendor:my-cpp|reference"`

> 入口函数必须是 `register` / `vllm_fl_register`（[文档与代码差异](known-issues.md)）。

### 第 3 步: 注入真实推理（L4）

1. `VLLM_FL_PLUGIN_MODULES=<你的插件>` 让每个 vLLM 子进程自动发现
2. `VLLM_FL_PER_OP` 钉住目标算子，其余算子保持默认
3. 基线/插件双跑同 prompt 同 seed
4. 断言（见[断言策略](testing.md#assertions)）:
   - 数值恒等实现 → 逐 prompt 前缀完全一致（前缀按[黄金](testing.md#golden)自适应）
   - 自定义数值实现 → 前 2 token 一致率 ≥ 2/3×N
5. 叠加[黄金回归](testing.md#golden)做跨会话漂移检测

### 第 4 步: 跨层一致性与收尾

- `run.py --consistency`: 同输入张量在 L0 直调 / L2 dispatch /
  参考实现处两两比对（张量级，容差 bf16=1e-2）
- `scripts/gen_report_scaffold.py` 生成[开发报告](reporting.md)骨架
- `scripts/report.py` 输出全矩阵汇总

## Triton 路线的差异

A1/A2 路线的全链路与之同构，差异仅在:

| 环节 | C++（B） | Triton（A1/A2） |
|---|---|---|
| kernel 编写 | csrc + pybind11 | `@pointwise_dynamic + @triton.jit` |
| L2 注册 | vendor backend + OpImpl(VENDOR) | A1: aten `Library.impl()`；A2: dispatch 插件双后端 |
| L4 注入 | PLUGIN_MODULES + PER_OP | A1 额外需 [sitecustomize 跨进程桥](route-a1-aten.md) + FlagGems 黑名单 |

Triton 同算子贯穿的自动化验证见
[tests/kernel_level/test_consistency.py](../tests/kernel_level/test_consistency.py)。

## 常见坑（全链路视角）

1. vLLM v1 前向在**子进程**——主进程注册不传播（A1 必须 sitecustomize）
2. 自研 kernel 与[黄金](testing.md#golden)路径不同时，输出会混沌分叉
   （随机权重放大数值微差）——不是 bug，用一致率断言
3. 性能基准长循环触发分配器池增长（0.04ms→16ms 失真）
4. 计数文件要 pid 分片（TP 多 worker 并发写竞争，否则计数低估 ~35x）
