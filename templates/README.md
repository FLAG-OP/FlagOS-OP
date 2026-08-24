# 模板目录

| 模板 | 用途 | 使用方式 |
|---|---|---|
| [op-development-report.md](op-development-report.md) | 算子开发报告 | 手写，或用脚手架自动生成骨架 |

## 自动生成报告骨架

```bash
python3 scripts/gen_report_scaffold.py --op my_op --route a2 --device p800-kunlunxin
# 产出 reports/my_op_p800-kunlunxin_report.md（环境/验证结果自动填好，
# 算子定义/实现/性能等留 TODO 手填）
```
