# 报告与环境快照

[← 返回文档中心](index.md)

## 组成

| 组件 | 位置 | 用途 |
|---|---|---|
| 报告模板 | [templates/op-development-report.md](../templates/op-development-report.md) | 手写完整报告的骨架（7 章+附录） |
| 环境快照 | `scripts/env_snapshot.py` | 自动收集环境配置章节 |
| 骨架生成器 | `scripts/gen_report_scaffold.py` | 一键生成半自动报告 |

<a id="scaffold"></a>
## 一键生成报告骨架

```bash
python3 scripts/gen_report_scaffold.py --op <算子名> --route <a1|a2|b> --device <profile>
# 产出 reports/<算子>_<profile>_report.md
```

自动填充:
- [环境配置](#env)整章（硬件 smi / 软件栈版本 / profile 全文 / 关键环境变量）
- 9 格矩阵中该路线 3 层的验证状态与耗时（读 `results/`）

`[手填]` 占位部分: 算子定义 / 实现说明 / 各层明细结论 / 性能数据 /
风险清单 / 结论。

<a id="env"></a>
## 单独收集环境快照

```bash
python3 scripts/env_snapshot.py --device <profile>            # 打印
python3 scripts/env_snapshot.py --device <profile> --out env.md
```

采集内容: OS/内核 · 设备 smi 输出（按 profile vendor 选工具） ·
Python/PyTorch/vLLM/FlagGems/vllm-plugin-FL/Triton/transformers 版本 ·
[设备 profile](device-profiles.md) 全文 · 关键环境变量
（CUDA_VISIBLE_DEVICES / VLLM_FL_* 等）。

无法自动采集的项（拓扑映射、厂商库版本、容器限制）标注 `[手填]`。

## 报告章节结构

```
1. 环境配置        ← 自动（本页工具）
2. 算子定义        语义 · PyTorch 参考实现 · 接口 · 数值规格
3. 实现说明        路线选择理由 · kernel 要点 · 注册与分发
4. 验证结果        ← 状态自动填; 明细手填（3 层 + 一致性 + 黄金）
5. 性能            本实现 vs 参考 vs 厂商原实现
6. 已知问题与风险
7. 结论与后续
附录               复现命令 · 产物路径
```

## 与全链路开发的关系

[全链路指南](fullstack-guide.md)的第 4 步即报告收尾——开发完成的
自然出口。`reports/` 已 gitignore（per-op 生成物不入库），
`templates/` 入库。
