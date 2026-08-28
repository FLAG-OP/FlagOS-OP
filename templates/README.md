# 模板中心

## 三件套

| 模板 | 用途 | 使用方式 |
|---|---|---|
| **[operator/](operator/)** | **算子样板**——单算子自包含: 三级 kernel（torch/Triton/硬件级）· 三层测试 · 黄金数据（声明式规格+生成脚本）· 精度/性能脚本 · REPORT 总报告+分册 | `cp -r templates/operator <你的算子名>/` |
| [op-test-report.md](op-test-report.md) | 测试报告（聚焦验证结果） | 复制填写；已填样例见 [reports/examples](../reports/examples/) |
| [op-development-report.md](op-development-report.md) | 开发报告（完整开发叙事，7 章） | 手写，或用脚手架自动生成骨架 |

样例报告（真实数据）: [gelu_and_mul 测试报告](../reports/examples/gelu_and_mul_test_report.md)
· [b-fullstack 开发报告](../examples/b-fullstack/report.md)

## 自动生成报告骨架

```bash
python3 scripts/gen_report_scaffold.py --op my_op --route a2 --device p800-kunlunxin
# 产出 reports/my_op_p800-kunlunxin_report.md（环境/验证结果自动填好，
# 算子定义/实现/性能等留 TODO 手填）
```
