# 开发报告: native_layer_norm（issue #55, 2026-11-25/26）

## 1. 语义定义
LayerNorm 私有原语（台账 13,593 次; 与 layer_norm 配对验证）。参考实现 `reference.py`（fp32 内部精度 + cast 回,
docs/acceptance.md 参考实现约定）; 判卷口径见 accuracy.md。

## 2. 实现选型
原生委托（3 返回值; native_layer_norm_backward 反向）。选型依据（真机探针 probes/thead_dense_p0_probe.py, 卡6）:
- 原生可用性与精度取证 → 复用/自研裁定;
- redispatch 键口径: **后端键 CUDA** 与直调逐位一致;
  CompositeExplicitAutograd 落 ATen 组合分解, 半精度分化（bf16 LN 差 0.031 实证）;
- 注册键口径: **AutogradCUDA**（CUDA 基键注册会覆盖该栈 autocast 粘合并
  致同键 redispatch 重入——RecursionError 实证）。

## 3. dispatch 证据链
注册 `aten::native_layer_norm@AutogradCUDA`; op 层拦截/回退/反向三段验证全绿
（intercepted_calls 见 test-report.md）。

## 4. 测试设计
三层覆盖要点: 非 BLOCK 整倍数维度（3072/5000/5120/4097/1023, #15a 教训）、
三 dtype、哨兵（确定性 + 输入敏感）、边界输入、反向对拍（vs fp32/CPU 参考）。

## 5. 黄金与精度
CPU 权威黄金（seed 可复现 + sha256 索引）; 官方 CPU ATen 语义对拍内建于
gen_golden; 真机校验结果见 accuracy.md。

## 6. 性能
短采样 + synchronize 口径; native/direct/reference 三方对照,
数字见 performance.md（可由 script/bench_perf.py 复现）。

## 7. 交付物
README / REPORT / PLATFORM / reports 四件套 / goldendata（spec+README,
data 不入库）/ script 三件套 / test 三层 / example 一键 / probes 取证。
