# 算子测试报告: gelu_and_mul（已填样例）

| 项 | 值 |
|---|---|
| 算子名称 | `gelu_and_mul` |
| 实现路线 | **A2** FlagOS 融合算子 |
| 开发级别 | **Triton 级** |
| 目标设备 | `p800-kunlunxin` |
| 测试日期 | 2026-08-28 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 是否覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ✅ | `run.py --route a2 --level kernel` | PASS（4.3s，metrics 落盘） |
| 框架层（注册/钉选/拦截） | ✅ | `run.py --route a2 --level op` | PASS（7.9s） |
| 应用层（真实推理） | ✅ | `run.py --route a2 --level framework` | PASS（108s，双跑+黄金） |
| 跨层一致性 | ✅ | `run.py --consistency` | PASS（4×4 全零误差） |
| 性能回归门禁 | ✅ | `scripts/perf_compare.py` | OK（19 用例 0 FAIL/WARN） |

## 2. 环境

共享镜像版本锁定核对通过（`scripts/check_env.py`）:
torch 2.9.0+cu129 · triton 3.0.0+03c4c9be · flag_gems 4.2.1rc0 ·
vllm 0.13.0 · vllm-plugin-fl 0.1.0 · transformers 4.57.1 · python 3.10.18。
完整快照见 `scripts/env_snapshot.py --device p800-kunlunxin`。

## 3. 精度结果

口径: 带 gate 乘法放大 → **相对误差**（`scripts/accuracy_report.py`），
参考 = 同输入 fp32 计算后 cast 回输入 dtype。

| dtype | 容差 | 自研 max rel err | 判定 | FlagGems† | 原生 |
|---|---|---|---|---|---|
| bf16 | 5e-2 | 3.02e-03 | ✓ | 2.78e-03 ✓ | 3.12e-03 ✓ |
| fp16 | 5e-2 | 3.69e-04 | ✓ | 3.74e-04 ✓ | 4.78e-04 ✓ |
| fp32 | 5e-2 | 6.02e-08 | ✓ | 1.00e-04 ✓ | 1.03e-04 ✓ |

† FlagGems 无该融合算子，对照为其 `gelu` 单算子组合。

哨兵: 确定性 ✓（同输入两次位级一致）· 输入敏感 ✓（x+1 输出变化）。

## 4. 性能结果

shape 8192×8192 bf16，短采样 20+100，子进程隔离采集:

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton 融合 | **0.052ms** | 7.67 TB/s | 1.00x |
| PyTorch 分解参考 | 3.69ms | — | 0.014x（自研快 **70x**） |
| FlagGems | ✗ 不可用（[known-issues #13](../../docs/known-issues.md)） | — | — |

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | FlagGems gelu(tanh) 在本栈链接失败 | 无同语义生产基线 | 已记录 #13 |
| 2 | XPU 尾块归约陷阱（本算子 shape 恒为 2 的幂，未触发） | 换非整倍数 shape 需防护 | 模板已内置 pad，#15a |

## 6. 结论

三层验证全绿、精度三方最优之一、相对分解路径 70x，**可合入**。
后续方向: FlagGems 修复 #13 后补生产基线对比；关注 #15a 在动态 shape
场景的防护是否需要下推到本算子。
