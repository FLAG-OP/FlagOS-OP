# matmul：XPU 2 首次完整性能基线

日期：2026-10-14。物理 XPU 2，个人容器 `tyanfeng-flagos-p800-lock`；
`CUDA_VISIBLE_DEVICES=1,2`、逻辑 `cuda:1`、`XMLIR_MATMUL_FAST_MODE=0`。
完整结论见 [性能报告](../performance.md)。

## 数据含义

- `workloads.json`：12 个非空形状 × 3 dtype 的 36 组输入定义和固定种子。
  每组测 forward/backward、triton/native，共 144 条记录。
- `baseline.json`：首次完整采集。每条用例独立进程，20 次预热、100 次调用
  取平均，计时前检查参考精度；反向同时计算和读回 dA、dB。
- `verification.json` / `gate.md`：完整 144/144 条独立复测及按仓库默认
  阈值计算的对比。结果为 OK 136 / WARN 4 / FAIL 4；8 条超阈值项均原生，
  自研 Triton 72 条均 OK。门禁结论保留为 FAIL。
- `anomaly_repeats.json` / `anomaly_repeats.md`：对 8 条原生异常各追加
  3 次复核。结果显示单次运行有明显波动，不能把异常归因于自研实现，也不
  足以覆盖原始门禁判定。
- `results.md`：完整基线轮的 36 行前后向自研/原生对照，单轮数字不代表
  稳定加速；不是未完成的验证轮结果。
- `provenance.json`：本次 Python 源码、输入规格、profile、环境锁与公共
  性能脚本的 SHA256，附设备映射及编译配置。工作树存在未提交修改，
  不能单靠记录中的 Git HEAD 识别本次实现。

## 口径限制

当前前向不要求梯度；自研反向直调数学反向，原生反向经 autograd 引擎，
原生前向图在计时外准备并复用。这些是调用延迟，包含分配与标量读回，
不等同于纯 kernel 时间，也不等同于 A1 应用中一次完整训练步骤。
基线仅适用于本次 P800 环境和用例定义。日常回归不可自动覆盖基线。
