# Changelog

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
