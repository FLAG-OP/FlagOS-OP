# 平台绑定清单（Platform Binding）— sdpa_math

> 当前交付包含两个平台：**ascend910**（自研 Triton：两段式 probs+PV，
> 小形状单 kernel 融合）与 **p800-kunlunxin**（vendor O/LSE + exact-P
> + A1）。cpu 为 ATen 组合对照。
> 绑定结论移植时逐项重验；详细实验数据见 `reports/`。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| fp32 语义参考 | `reference.py` |
| 黄金规格与数据 | `goldendata/`（CPU 生成） |
| 三层测试、性能脚本 | `test/`、`script/`（位置参数传平台） |
| A1 注册（含平台分发） | `register.py` |

## 2. ascend910 绑定（自研 Triton）

- A1 拦截 key **AutogradPrivateUse1**（sdpa 同结论）
- Triton tile 上限 64×64（单核 192KB UB——见 sdpa
  reports/perf_analysis.md，本算子同栈约束）
- 与 sdpa 全量版的关系：本算子是 `_scaled_dot_product_math` 的
  exact-P 分解路线；形状路由建议沿用 sdpa `kernel/auto_dispatch.py`
  的截流范式（大 S 走原生）

## 3. p800-kunlunxin 绑定

- XMLIR 呈现 CUDA 张量（AutogradCUDA）
- vendor O/LSE 委托 + exact-P 补偿；`kernel/p800_fast_level.py`
- probes/ 归档（7 个探针）为该平台的证据链

## 4. 新平台接入

参照 `ops/sdpa/MERGE.md` 七步流程（本算子暂无独立 MERGE.md，结构
对齐后可裁剪复用）；kernel 本体 + register 平台分发 + 三层验证 +
包化 `__init__.py`（已有）四件套不可省。
