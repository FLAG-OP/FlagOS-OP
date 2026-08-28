# 算子样板（copy-paste 起点）

开发一个新算子，从这里开始:

```bash
cp -r templates/operator <你的算子名>/
cd <你的算子名>/
grep -rl my_op . | xargs sed -i 's/my_op/<你的算子名>/g'
# 逐个把 <TODO> 替换为你的实现
```

## 目录结构

```
<你的算子名>/
├── README.md            本说明
├── REPORT.md            算子总体报告（一页看全，链到分册）
├── reference.py         fp32 语义参考——判卷标准，先写它
├── register.py          三路线注册（A1/A2/B 三选一）
├── kernel/              三级实现
│   ├── torch_level.py         torch 级（ATen 组合）
│   ├── triton_level.py        Triton 级（内置 #11/#15 防护）
│   └── hardware_level/        硬件级（类 C）
│       ├── kernel.cu              CUDA C++（NVIDIA；P800 执行 ✗ #12）
│       ├── xtorch_binding.cpp     厂商 C++ 绑定（P800 当前可跑）
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
└── reports/             分册报告模板
    ├── accuracy.md            精度详细报告
    └── performance.md         性能详细报告
```

## 开发顺序

1. **先写 `reference.py`**——语义没定清楚之前不要写 kernel
2. 补 `goldendata/inputs_spec.yaml` 并
   `python3 script/gen_golden.py --device cpu`——黄金先行，任何实现
   出来就有判卷标准
3. 实现三级 kernel 中需要的级别，
   `python3 test/kernel_level.py --device <profile>` 全绿
4. `python3 script/check_accuracy.py --impl <级别> --device <profile>` 过黄金
5. 选路线（`register.py` 三选一）→ `test/op_level.py` → `test/framework_level.py`
6. `script/bench_perf.py` 看性能；挂 `common/perf_registry.py` 进回归门禁
7. 填 `REPORT.md` 总报告 + `reports/` 分册；已填样例见
   [reports/examples](../../reports/examples/)

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
