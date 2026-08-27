# Changelog

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
