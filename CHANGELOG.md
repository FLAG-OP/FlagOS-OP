# Changelog

## [0.8.2] - 2026-08-28

### Changed
- 硬件级模板改为占位符 + 文字说明形态（模板库定位: 骨架与指引，
  不预写实现）: kernel.cu 保留结构要点（网格划分/访存/精度/尾块提示），
  xtorch_binding.cpp 标注厂商 API 位置与 #2/#5 实测坑

## [0.8.1] - 2026-08-28

### Changed
- 算子样板按"单算子自包含"重组（采纳用户建议结构 + 相关工作对标）:
  kernel/（三级实现: torch/Triton/硬件级类 C）· test/（三层测试）·
  goldendata/（声明式规格 + 黄金数据）· script/（生成/精度/性能脚本）·
  REPORT.md（总体报告）+ reports/（精度/性能分册）

### Added
- 硬件级模板: kernel.cu（CUDA C++，标注 #12 限制）+ xtorch_binding.cpp
  （P800 当前可跑的厂商 C++ 绑定，含已验证 include 三链）+ BUILD.md
- goldendata/inputs_spec.yaml: 维度/精度/数量/随机分布/特殊功能
  （zeros/large/boundary）声明式规格（参考 KernelBench problem 声明）
- script/gen_golden.py: 按规格生成黄金输入+参考输出+sha256 索引
  （CPU 实测 24 组生成 ✓）
- script/check_accuracy.py: 读黄金按容差判定（实测 24/24 pass ✓）
- script/bench_perf.py: 单算子性能测试（复用 common.perf 口径）
- REPORT.md 总体报告 + reports/accuracy|performance.md 分册模板

## [0.8.0] - 2026-08-28

### Added
- 算子样板 templates/operator/（7 文件 copy-paste 起点 kit）: 语义参考 /
  Triton kernel（内置 #11 device 上下文与 #15 尾块 pad 两条硬约束）/
  kernel 层测试（metrics 落盘约定）/ 三路线注册 / op 与应用层测试骨架 /
  性能回归用例
- 测试报告模板 templates/op-test-report.md（测试范围/精度/性能/风险/结论）
- reports/examples/ 入库: gelu_and_mul 已填测试报告（真实数据）+ 2 份
  开发报告骨架 + 索引（原 reports/ 整体 gitignore，报告样例不可见）

### Changed
- 根 README 新增"仓库分区"表（样板/正式实现/教学样例/自动化/文档五区）
- 全链路指南第 0 步、CONTRIBUTING、architecture 目录树接入样板入口
- .gitignore: reports/* 但保留 reports/examples/

## [0.7.0] - 2026-08-28

### Added
- scripts/accuracy_report.py: 自研/FlagGems/原生 三方同输入同参考精度对比
- 样例索引新增"精度速览"表: 自研 15/15 全过；FlagGems softmax 三 dtype
  全超差（fp32 1.5e-2）——其性能优势为低精度换取

### Fixed
- ⚠️ softmax 尾块正确性 bug（#15a，由精度探针发现）: N 非 BLOCK 整数倍时
  masked load+tl.sum 污染结果（三种防护无效）；修复为 pad 到 2048 倍数
- ⚠️ 移除 @triton.autotune（#15b）: 本栈 autotuner 选出非法 num_warps=5，
  同 kernel 不可复现地时对时错；改固定 BLOCK_N=2048
- softmax 测试形状补非整倍数 N（3072/5000/5120）——原测试全整倍数漏测

## [0.6.2] - 2026-08-28

### Added
- FlagGems 生产基线接入性能对比与回归体系: softmax / bmm 直调，
  silu_and_mul 用 FlagGems silu 单算子组合（无该融合算子）
- known-issues #13（FlagGems gelu(tanh) 链接失败，XPU libdevice
  tanh → "Unsupported"）与 #14（编译错误被 NameError:sys 掩盖）
- common/xpu_compat.py: 经 backends 注册表给隐藏的 xpu compiler
  第二实例注入 sys，编译失败暴露真实原因

### Changed
- bmm/softmax/b-fullstack/hw 样例的性能段输出 FlagGems 基线行（防御式）
- A1/A2 因 #13 无同语义 FlagGems 基线，文档注明而非用 gelu(none) 冒充
- 样例性能速览表增加 FlagGems 基线列（softmax 上 FlagGems 反超自研）

## [0.6.1] - 2026-08-27

### Fixed
- 文档-代码矛盾: 4 处仍写 run() 只返回 bool，与 0.6.0 的 dict 约定不一致
- b-kernel 性能段静默吞异常（perf=null 无原因）→ 打印 SKIP 原因
- perf_compare 常驻 "NEW 1" 噪音: intake 用例单独列出，不参与门禁与 NEW 统计
- b-kernel 负例 flaky: gelu_tanh_and_mul 缺陷偶发（#5），"负例未复现"
  误判整格 FAIL → 降级 WARN，通过不再依赖坏 kernel 每次都坏
- p800 profile 该负例的 known-issues 引用误写 #10 → #5

### Changed
- device-profiles 补冒烟模型要求（结构/权重格式/词表与 tokenizer）

## [0.6.0] - 2026-08-27

### Added
- 环境锁定: configs/env/p800-kunlunxin.lock.yaml（共享镜像实测版本）
  + scripts/check_env.py 核对（--require-model 一并检查模型路径）
- 模型路径泛化: 环境变量 FLAGOS_MODEL_PATH 覆盖 profile，换机器不改 YAML
- tests/unit/ 基础设施单测 20 个（intake 契约 / perf 记录与基线合并 /
  门禁分级 / KernelSpec 适配器与哨兵 / profile 覆盖），CPU 即可运行
- CI 新增 unit-tests job（CPU torch + pytest），从纯结构检查升级为跑真实测试

### Changed
- run() 约定支持返回 {"ok": bool, ...指标}，结果 JSON 新增 metrics 字段
- kernel 层 a1/a2/b 测试落盘精度 max_err / 哨兵结论 / 性能延迟
- 报告脚手架自动渲染 metrics（第 4 章不再全部手填）
- perf_compare 门禁分级抽为 classify_delta 纯函数（可单测）
- kernel_spec 哨兵检查的 synchronize 加可用性保护（CPU 单测可用）

## [0.5.6] - 2026-08-27

### Fixed
- b-fullstack 三处历史重命名残留: "kernel 层 层"/"op 层 层"（README+docstring）
  与 "厂商语言 kernel/厂商语言路线" 误称（该样例是 torch 级 C++ 调 ATen，
  路线名为厂商算子注册）

## [0.5.5] - 2026-08-27

### Changed
- 消除模板句口癖（"读这一页能得到什么"×4、"典型开发级别:"引言×3、
  "## 定位"标题），全部改写为自然连贯的说明段落
- 根 README 开篇从名词堆叠改写为两段完整叙述（库解决什么问题、
  3×3 矩阵如何组织）
- 路线文档把"级别与路线正交"从事故在引言里的标签，改写成
  适用场景内的正文段落，并删除文末重复的级别标签行
- docs index 补一段文档分层说明（主线/专题/参考）；样例索引开篇
  改写为可执行指引

## [0.5.4] - 2026-08-27

### Changed
- architecture/testing/reporting 补定位开篇句（与其余 10 篇风格对齐）
- testing 矩阵表归入"总览"小节（消除目录后表格悬空）
- 样例索引回链全链路指南覆盖矩阵（样例视角 ↔ 步骤视角互达）
- index 新手线注明性能/AI 生成走专题线

## [0.5.3] - 2026-08-27

### Fixed
- known-issues 错误引用 3 处: swiglu 不写输出是 #2（route-b/b-op/b-framework 误写 #5）
- route-a1 错乱文本（"torch torch 算子替换"、"典型 kernel 层级: Triton（Triton）"）
- route-b audit 段重复句、"kernel 层级"→"开发级别"
- route 三篇 breadcrumb 残留旧路线名尾巴
- device-profiles: 循环引言重写、"字段速查"空壳章节（表归位）
- getting-started "3 步接入" 与 device-profiles "4 步" 不一致
- reporting: 全链路步骤编号 4→6（指南重构后）

### Changed
- route-a1 样例补 bmm/softmax（A1 全链路）；route-b 样例补 b-kernel/b-fullstack/hw-kernel
- routes README 标题统一正式路线命名
- 新增 tests/kernel_level/README.md（与其他两层测试对称）

## [0.5.2] - 2026-08-27

### Changed
- 全链路指南重构: 从"C++ 单路线 4 步说明"改为"七步抽象流水线"
  （每步: 抽象动作 + 完成标志 + 分级/分路线变体 + 范本映射）
- 新增"步骤 × 样例覆盖矩阵"（14 样例 × 7 步）与"按需求快速选范本"表
- 修正 a1-op README 重复词（"torch torch"）

## [0.5.1] - 2026-08-27

### Fixed
- 根 README 已知问题计数 11→12
- architecture: 全链路样例列表去重（softmax/backward 重复两次）
- architecture: 路线图补 AI 生成源旁路（与根 README/docs index 三处一致）

### Changed
- architecture 目录树同步 intake/ 与 perf/ 目录、14 样例数、perf harness
- docs index 阅读路径新增"性能"与"生成"两条线
- testing/getting-started 挂接性能回归与 intake 产物入口
- testing/perf/intake/reporting/device-profiles/known-issues 补齐"下一步"导航链
- known-issues #10 补短采样不足的实测证据与子进程隔离防护；#11 补 intake 负例自动拦截

## [0.5.0] - 2026-08-27

### Added
- 性能回归追踪: 统一基准 harness（common/perf.py）+ 用例注册表 + 15 个首批用例（6 样例 + kernel 层矩阵格）
- perf_run / perf_compare 脚本: 结构化运行记录（results/perf/runs/）+ 入库基线（perf/baselines/）+ 双档门禁（慢 20% WARN / 慢 30% FAIL）
- AI 生成算子 intake 通道: manifest 契约（JSON Schema）+ 三级验证脚本 + KernelGen 正/负示例 case
- 负例 kernelgen-gelu-no-device-context: 哨兵检查在 intake 阶段拦截 known-issues #11 类静默 no-op
- 文档: 性能回归追踪 / AI 生成算子接入（含生命周期图与阈值语义）
- CI: intake 契约校验 + 性能基线结构检查

### Fixed
- examples/README 性能速览表错行（BMM 行链到 softmax，混入 3 行样例表内容）

## [0.4.0] - 2026-08-27

### Added
- 硬件级开发样例 hw-kernel-example: xtorch_ops 厂商原语组合 + CUDA C++ 参考
- SDK 就绪模板 sdk_template/: XPU kernel 源码 + 编译指南 + 自动检测
- reduction 样例 softmax-fullstack（流式三遍归约 + @triton.autotune）
- backward 样例 backward-example（autograd fwd+bwd + 训练冒烟）
- 样例定位图: 14 个样例映射物理栈层 × 路线
- 物理栈架构图: 应用→框架→编译→算子库→硬件 五层垂直视图

### Changed
- 路线名: aten dispatcher→torch 算子替换 / FlagOS dispatch→FlagOS 融合算子 / vendor backend→厂商算子注册
- 开发级别: FW/TR/HW→torch 级/Triton 级/硬件级
- 验证层级: L0/L2/L4 编号→物理栈层名（算子库层/框架层/应用层）
- 体系图: 四子图平行结构→物理栈垂直视图
- 根 README: 139→61 行（着陆页标准）
- 全文档加"下一步"导航（6 个文档形成阅读链路）
- 核心心智模型在需要处重复出现（不再只靠索引跳转）

### Fixed
- known-issues #12: CUDA C++(NVIDIA) 无法在 P800/XPU 执行
- 报告模板与脚手架同步物理栈视角

## [0.3.0] - 2026-08-26

### Added
- 开发三级重构: torch 级 / Triton 级 / 硬件级
- 样例定位图: 13 个样例直接映射 3×3 矩阵与开发层级
- 算子开发报告模板 + 环境快照 + 骨架生成器
- BMM 全链路样例（Triton 路线）+ 发现 known-issues #11
- CI 工作流（markdown 链接检查 + Python 语法）
- CONTRIBUTING.md / LICENSE (Apache-2.0)

### Fixed
- known-issues #11（裸 Triton 缺 device 上下文→静默 no-op）
- 路线名与开发层级解耦（消除正交性矛盾）
- 跨层一致性结果落盘 / TP 计数文件并发写竞争

## [0.2.0] - 2026-08-25

### Added
- 跨设备黄金输出（CPU transformers / 加速卡 vLLM）
- 漂移实验工具 + 逐 prompt 自适应断言前缀
- 全链路样例 b-fullstack（C++ 路线）+ 开发报告
- 测试输入声明式模板 + token 钳制（规避 embedding 越界）
- 文档体系: docs/ 11 篇中文文档 + 交叉链接 + Mermaid 图

## [0.1.0] - 2026-08-24

### Added
- 初始版本: 3 路线 × 3 层级 = 9 格矩阵 + 跨层一致性
- 设备 profile 芯片泛化（P800 参考实例）
- 厂商 kernel 直测 + C++ JIT 编译闭环 + 哨兵检查
- 11 条已知问题（含 swiglu 不写输出等重大发现）
