# 探针与调试脚本（开发过程证据链）

这些脚本不在交付主链上，但保留它们是因为报告中的关键结论都由其产出。
按用途分组:

## 基线探测（开发报告§4.3 误归因记录的原始依据）

- `probe_baseline.py/.log` — 原生/flag_gems enable/直调三路可用性扫描
- `probe_mask_fail.py/.log` — "mask 编译失败"误归因的完整 traceback
  （实为 flag_gems rand kernel UB 溢出）
- `probe_mask_v2/v3.py/.log` — mask 语义修正（bool 保持遮蔽语义）后的重测
- `probe_register.py` + `register_probe.log` — **A1 注册点证据链**:
  dispatch dump 显示 PrivateUse1 注册 active 但 call count=0
- `debug_reg.py/.log`、`debug_reg2.py/.log` — 定位到 AutogradPrivateUse1
  拦截点 + autograd 破坏问题
- `debug_extreme.py/.log` — extreme fp32 误差解剖（原生同误差 3.381e-4
  的互咬合证据）
- `debug_native_acc.py/.log` — native shim 因果折叠的必要性验证
- `smoke.py/.log` — kernel 首次冒烟
- `_soc.py`、`_inspect_dot.py` — 芯片代次/接口签名查证
- `guard_check.py/.log` — 平台三层守卫验证（元数据/调用拦截/注册正向）
- `doc_consistency.py` — 文档一致性终检（链接/引用/旧数据残留，
  11 份文档全过；文档改动后重跑此脚本）
