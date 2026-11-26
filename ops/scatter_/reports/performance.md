# P800 性能（2026-11-26）

使用 common.perf.run_case：20 次预热、100 次测量，前后同步 cuda:0。
fp32、最后一维逆序 scatter_.src，源和目标预分配，不计初始化成本。
计时覆盖校验和 Python 调用，不仅是设备 kernel。

| shape | Triton ms | native ms | native/Triton |
|---|---:|---:|---:|
| 7×1031 | 0.273763 | 0.017000 | 0.062x |
| 128×4096 | 0.959844 | 0.048540 | 0.051x |

原始记录见 `perf-p800.json`。本次实现明显慢于原生，不建议作为性能替换。
为确保正确性，XMLIR 需要尾部标量程序、索引范围检查及 Host 同步；
stride 解码与 int64 地址也增加开销。未隐藏校验成本或删除不利结果。
将大整块从 256 调到 16384 曾改善大尺寸耗时，但仍不足以加速原生。
后续方向：平台可靠的设备断言、尾块 scatter lowering 修复、连续布局专核。
四条记录入库 `perf/baselines/p800-kunlunxin-gems5.json`，仅为本算子基线，
不代表 gems5 的全库性能矩阵。
