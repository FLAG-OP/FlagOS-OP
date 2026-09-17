# sdpatten-op: aten::scaled_dot_product_attention 的 Triton 级实现（Ascend 910）

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
组织结构对齐 `templates/operator` 与 softmax-fullstack 范本。

## 一页看全

| 项 | 值 |
|---|---|
| 路线 / 级别 | **A1**（aten 拦截，注册点 `AutogradPrivateUse1`）/ **Triton 级** |
| 平台 | **ascend910 专属**——绑定项清单与移植指引见 [PLATFORM.md](PLATFORM.md) |
| 多平台 | 第二平台实现后按 [MERGE.md](MERGE.md) 合入（黄金/测试/注册链复用，旧引用零改动） |
| 语义 | softmax(QKᵀ·scale + mask)·V · GQA · causal · bool/float mask · fp32 内部 |
| 验证 | kernel 层 37/37 · 黄金 265/265 · 框架层拦截+梯度 ✅ · 应用层 mini-decoder 双跑 ✅ |
| 性能 | vs FlagGems Triton SDPA **3.3-17.6x 快**；vs 原生 CANN 1.9-10.8x 慢（根因: 栈 GEMM 5.2x × UB tile 上限，见[性能分册](reports/performance.md)） |

## 目录

```
sdpatten-op/
├── REPORT.md                 总报告（一页看全 + 交付物清单）
├── README.md                 本文件
├── PLATFORM.md               平台绑定清单（8 项 Ascend 绑定 + 移植指引 + 引用防误用）
├── MERGE.md                  多平台合并指南（七步流程 + 选择器 + FlagGems 上游路径）
├── requirement.md            原始需求
├── reference.py              fp32 语义参考（判卷标准，手写不调 F.sdpa）
├── register.py               A1 注册（autograd.Function 包装 + 注册守卫 + 计数）
├── example.py                一键三层复跑
├── _profile.py               本地设备 profile（ascend910）
├── kernel/
│   ├── triton_level.py       主实现（online-softmax two-pass + causal 截断
│   │                         + PLATFORM 元数据 + 调用守卫）
│   ├── torch_level.py        ATen 组合（第二判卷人）
│   └── _native_shim.py       原生对照（NPU causal+mask 并存的折叠适配）
├── test/
│   ├── kernel_level.py       精度 36 组 + 哨兵 + 快速性能
│   ├── op_level.py           拦截命中 / 逐位一致 / 梯度 vs 原生子进程
│   ├── op_phase2_native.py   原生梯度采集（独立进程）
│   └── framework_level.py    应用层（mini-decoder 双跑 + 行为一致性）
├── goldendata/               黄金（inputs_spec.yaml + data/ 265 组 + index）
├── script/
│   ├── gen_golden.py         黄金生成（双向互验）
│   ├── check_accuracy.py     精度判定（--impl triton|native|reference）
│   ├── bench_perf.py         三方性能对照
│   ├── perf_explore.py/.2    根因: tile 扫描 + matmul 天花板 + 带宽核算
│   └── perf_variants.py      写法变体单项 A/B（V1-V5）
├── probes/                   开发过程证据链（12 探针 + 日志 + guard_check）
└── reports/                  交付四件套 + perf_analysis 根因专项 + perf json
```

## 快速开始

```bash
python3 example.py                    # 三层一键复跑
python3 test/framework_level.py        # 应用层单独跑
python3 test/kernel_level.py          # kernel 层
python3 test/op_level.py              # A1 拦截 + 梯度
python3 script/gen_golden.py          # 黄金（CPU，265 组）
python3 script/check_accuracy.py --impl triton --device npu:0
python3 script/bench_perf.py --json-out /tmp/perf.json
```

## <a id="应用层"></a>应用层说明

本栈 vllm 0.20.2+empty 为空壳，无法注入真实推理引擎——按 FlagOS-OP
softmax-fullstack 的先例（"attention scores 消费 softmax"），应用层以
**轻量消费方**落地: `test/framework_level.py` 构建 mini-decoder
（Llama 风格 4 层 · GQA 8/2 头 · causal · fp16），业务代码只调
`F.scaled_dot_product_attention`（标准 aten 路径零改动），基线/注册
双跑断言: 拦截命中（count=28）· logits 数值一致（1.95e-3）·
贪心续写序列一致率 1.00。待本栈有可用 vllm 后可平移到真实引擎
（A1 注册经 sitecustomize 注入子进程）。

## 本栈关键发现（对上游有价值）

1. **torch_npu 的 A1 注册点**: PrivateUse1 永不命中（C++
   AutogradPrivateUse1 不 redispatch）；须注册 AutogradPrivateUse1 且
   用 autograd.Function 包装。证据链: `probes/debug_reg*.py` + dispatch dump
2. **FlagGems SDPA 未接入 aten 分发**: enable 后输出与原生 diff=0.0
   （requirement "未发现后端接入"的实测确认与机制解释）
3. **三条 Triton-Ascend 硬约束**: dot 强制同 dtype（P 须 cast）·
   fp32 默认 tf32（须显式 ieee）· exp2 慢路径（FA2 惯例负优化）
4. **causal 循环截断 1.93x**: 编译器无法从运行时 where 推断循环边界
   收缩——写法粒度决定 cube 利用率的实例
5. **平台防误引三层机制**: 元数据（`PLATFORM`）+ 调用守卫 + 注册守卫
   （`probes/guard_check.py` 全部实测）——函数名不改，防的是跨平台
   静默误用（[PLATFORM.md](PLATFORM.md) §5）

## 环境

Ascend 910_9382 ×8 · CANN 9.0.0 · torch 2.10.0+cpu · torch_npu 2.10.0 ·
triton 3.5.1 · flag_gems 5.3.5 · python 3.11.15（详见[开发报告](reports/development.md)§1）
