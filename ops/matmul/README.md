# matmul-op: `aten::matmul` A1 实现

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

当前交付状态为 **PARTIAL**。合并后的 CPU CI、包化导入和 P800 功能回归见
[PR 自查记录](reports/pr-readiness.md)；历史性能门禁遗留单独保留。

按 `templates/operator/` 开发。需求是 FP32/FP16/BF16 的向量、矩阵、批量及广播 matmul，前向和左右输入一阶反向分别验证。昆仑芯 P800 锁定镜像个人容器已在正式物理 XPU 2 完成黄金和三级功能测试，并建立 144 条性能基线；独立复测也已补齐到 144 条，门禁保留 4 FAIL / 4 WARN（均为原生路径）。2026-10-14 确认本阶段应用层以小型 framework 为准，真实模型及 tiny vLLM 不在本阶段范围。

## 一页看全

| 项 | 值 |
|---|---|
| 接口 | `torch.matmul` / `aten::matmul` |
| 接入 | A1 `AutogradCUDA`；实际命中计数 |
| 生产计算 | 自研 Triton `tl.dot`，32×32 输出 tile；保留动态 K 修复 P800 BF16 `K=1` 编译问题，默认统一走 dot |
| 反向 | 两次自研 matmul + 广播维归约 |
| 正确性 | CPU 参考生成 144 组；直调 Triton、A1 在物理 XPU 2 均对黄金 144/144；三级测试全绿 |
| 性能记录 | XPU 2：12 非空形状 × 3 dtype × 前后向 × 自研/原生，基线和独立复测均 144 条；门禁 4 FAIL / 4 WARN 均为原生路径，当前不进行性能优化 |

## 实现要点

`kernel/_shape.py` 统一处理向量升降维、批次广播和梯度归约。`kernel/triton_level.py` 统一使用 dot；外积/向量归约已迁至 [`probes/outer.py`](probes/outer.py)，通过 `matmul_outer(a, b)` 显式用于诊断，旧算法环境变量不再控制正式实现。P800 BF16 `K=1` 异常已定位到 `TritonSDNNCombineBefore` 融合阶段丢失输入 tile 清零，导致剩余 31 个 K 项读到 SRAM 残留；现用 `do_not_specialize=["K"]` 保留有效 K 掩码，撤掉自动外积回退。XPU 2 残留数据回归及黄金、三级测试均通过，见 [根因报告](reports/k1_root_cause.md)。入口仍通过运行时设备检测，仅在昆仑芯实化矩阵维零 stride（例如 `sum()` 反向的全 1 梯度）。调用方继续使用同一个 `matmul_triton`。高阶梯度不支持，其他厂商仍须分别验证。

`register.py` 对 P800 把 profile 的 `CUDA` key 提升为 `AutogradCUDA`；只注册 `CUDA` 会在 `aten::matmul_backward` 走错路径。`kernel/torch_level.py` 是语义诊断，不能算作自研 Triton 通过。应用层使用 FlagGems GELU 与两次 A1 matmul，独立进程运行三个测试层级。

## 个人容器复现

从 `/workspace/tyanfeng/FlagOS-OP` 运行，先确认 `xpu-smi` 中目标卡空闲。正式映射为 `CUDA_VISIBLE_DEVICES=1,2`，逻辑 `cuda:1` 对应物理 XPU 2：

```bash
python ops/matmul/script/gen_golden.py
python ops/matmul/script/check_accuracy.py --impl torch --device cpu
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul
```

性能测试固定 `XMLIR_MATMUL_FAST_MODE=0`，输入规格、计时口径与逐项结果见
[性能报告](reports/performance.md)。日常复测不加 `--update-baseline`。性能
异常和原生路径波动已留档，本阶段暂不进行性能优化。

物理 XPU 1 的 PyTorch 标量读回曾卡住，问题发生在 matmul 之前；`example.py` 先做 30 秒设备读回预检，并对每层设置 180 秒超时。`test/real_model.py` 和相关 vLLM 日志保留作历史探索记录，不属于当前运行流程或交付阻塞项。

查看 [需求](requirement.md)、[总体报告](REPORT.md)、[开发报告](reports/development.md)、[测试报告](reports/test-report.md)和[性能报告](reports/performance.md)。
