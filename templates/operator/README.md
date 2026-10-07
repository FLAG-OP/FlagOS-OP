# 算子样板（copy-paste 起点）

开发一个新算子，从这里开始:

```bash
cp -r templates/operator ops/<你的算子名>/
cd <你的算子名>/
grep -rl my_op . | xargs sed -i 's/my_op/<你的算子名>/g'
# 逐个把 <TODO> 替换为你的实现
```

## 目录结构

```
<你的算子名>/
├── README.md            本说明
├── __init__.py          包化入口（多算子同进程: from ops.<算子名> import ...）
├── REPORT.md            算子总体报告（一页看全 + 交付物清单）
├── example.py           一键编排（三层一次跑完）+ perf_cases 入口
├── reference.py         fp32 语义参考——判卷标准，先写它
├── register.py          三路线注册（A1/A2/B 三选一; A1 含平台分发
│                        与 dispatch_key→backend 映射）
├── kernel/              三级实现
│   ├── torch_level.py         torch 级（ATen 组合）
│   ├── triton_level.py        Triton 级（内置 #11/#15 防护 + PLATFORM
│   │                          元数据 + 调用守卫）
│   ├── backends/              平台 backend（第二平台出现时启用;
│   │     __init__.py          选择器范式见 sdpa 三平台实证）
│   ├── auto_dispatch.py       智能路由（可选: 大 shape 截流原生——
│   │                          sdpa e2e 实证生产链 0.99x 原生）
│   └── hardware_level/        硬件级（类 C）
│       ├── kernel.cu              CUDA C++ 骨架（占位 + 要点；P800 ✗ #12）
│       ├── xtorch_binding.cpp     厂商 C++ 绑定骨架（API 位置与坑已标注）
│       └── BUILD.md               编译指南（含已验证 include 三链）
├── test/                三层测试
│   ├── kernel_level.py        算子库层直测（精度/哨兵/性能 + metrics）
│   ├── op_level.py            框架层注册/钉选/拦截
│   └── framework_level.py     应用层（复用仓库 harness）
├── goldendata/          黄金数据
│   ├── inputs_spec.yaml       声明式规格（维度/精度/数量/分布/special）
│   └── README.md              字段说明（data/ 生成物不入库）
├── script/              工具脚本
│   ├── gen_golden.py          黄金生成（按规格，含 sha256 索引）
│   ├── check_accuracy.py      精度测试（读黄金 → 按容差判定）
│   └── bench_perf.py          性能测试（短采样 + 同步）
└── reports/             交付报告模板（四件套）
    ├── development.md         开发报告（7 章）
    ├── test-report.md         测试报告（范围矩阵）
    ├── accuracy.md            精度分册
    └── performance.md         性能分册
```

结构与已填范本 [softmax-fullstack](../../examples/softmax-fullstack/)
完全同构——模板是空壳，它是实肉；从本模板复制出的目录长成它的样子。

## 开发顺序

1. **先写 `reference.py`**——语义没定清楚之前不要写 kernel
2. 补 `goldendata/inputs_spec.yaml` 并
   `python3 script/gen_golden.py --device cpu`——黄金先行，任何实现
   出来就有判卷标准
3. 实现三级 kernel 中需要的级别，
   `python3 test/kernel_level.py --device <profile>` 全绿；
   三层就绪后 `python3 example.py <profile>` 一键跑完
4. `python3 script/check_accuracy.py --impl <级别> --device <profile>` 过黄金
5. 选路线（`register.py` 三选一）→ `test/op_level.py` → `test/framework_level.py`
6. `script/bench_perf.py` 看性能；把 `example.py` 的 `perf_cases` 登记
   进 `common/perf_registry.py`，进回归门禁
7. 填交付报告四件套（REPORT + development + test-report + 分册）；
   已填范本: [softmax development](../../examples/softmax-fullstack/reports/development.md)

## 设计参考（相关工作）

| 来源 | 借鉴点 |
|---|---|
| KernelBench | 声明式 problem 规格（shape/dtype 显式）→ `inputs_spec.yaml` |
| ONNX Runtime test-data | 输入/期望作为数据目录 + 哈希索引 → `goldendata/` |
| CUTLASS | 测试 harness 与数据分离 → `test/` 与 `goldendata/` 分开 |
| FlagGems | pytest 风格多 shape×dtype 精度断言 → `check_accuracy.py` |
| 本库实测 | #11 device 上下文 / #15 尾块与 autotune 两条硬约束内置 |

## 三条硬约束（违反必错）

1. **Triton 启动必须包 device 上下文**（[#11](../../docs/known-issues.md)）
2. **尾块归约要安全**: N 非 BLOCK 整倍数时 masked load + `tl.sum` 被污染，
   `other`/`tl.where` 均救不了——pad 或两阶段（[#15](../../docs/known-issues.md)）
3. **不要 `@triton.autotune`**: 本栈会选出非法 `num_warps=5`，
   不可复现（#15b）


## 与 ops/ 实际交付的对齐（2026-10 范式更新）

模板已同步 ops/ 三个新约定（实证来源标注）:

1. **包化**（`__init__.py`）: 多算子同进程时顶层名 register/kernel
   互相遮蔽——包态 `from ops.<算子名> import register_a1` 根治;
   单算子脚本模式不变（实证: sdpa/reports/e2e_mini_llm.md 三缺陷）
2. **平台分发**（register.py）: `register_a1(dispatch_key, counter,
   impl, platform)`; torch_npu 栈拦截点 **AutogradPrivateUse1**
   （PrivateUse1 永不命中——sdpa 开发报告 §3.1 dispatch 证据链）
3. **backends/ 选择器**: 第二平台出现且平台绑定清单结构性分叉时
   启用（范式: sdpa 三平台 kernel/backends/）; 单平台保持
   triton_level.py 顶层实现
4. **智能路由**（可选 auto_dispatch.py）: 大 shape 截流原生——
   生产链端到端 0.99x 原生（sdpa e2e 实证）; 须注意 aten 注册内
   转发原生会无限递归, 用函数层 patch（sdpa 教训）

参考样例: [ops/sdpa](../../ops/sdpa/)（三平台 + 路由 + e2e）·
[ops/embedding](../../ops/embedding/)（三平台 + 委托模式 + facade）。

## 交付自查清单（合并前逐项打勾）

> 2026-10-07 全库审计后新增——审计发现 13 算子缺包化入口、多平台
> 算子缺 PLATFORM/MERGE 文档。后续作者请勿再欠账:

- [ ] 核心八件: README / REPORT /  / reference /
      register / example / kernel/ / test/
- [ ] 三层测试: test/{kernel,op,framework}_level.py 全绿
- [ ] 性能对照: script/bench_perf.py（native 参照必备）+ reports/ 四件套
- [ ] 黄金: goldendata/ + script/check_accuracy.py 全过
- [ ] 包化回归: `ops/test_packaging.py` 全算子导入扫描通过
- [ ] 多平台时: PLATFORM.md（绑定清单）+ MERGE.md（七步流程）
- [ ] 署名: 改动处 `wt-YYYY-MM-DD-fix` + `# wt <邮箱>`，commit 作者一致
