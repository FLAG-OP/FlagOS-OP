# 模板中心

## 三件套

| 模板 | 用途 | 使用方式 |
|---|---|---|
| **[operator/](operator/)** | **规范算子样板**——单算子自包含: 三级 kernel、三层测试、黄金数据、精度/性能脚本和四件套报告 | 新算子从这里开始：`cp -r templates/operator ops/<你的算子名>/` |
| [op-test-report.md](op-test-report.md) | **兼容报告模板**（聚焦验证结果） | 仅用于旧目录或独立报告；应用证据须填写 C1/C2 |
| [op-development-report.md](op-development-report.md) | **兼容报告模板**（完整开发叙事，7 章） | 仅用于旧目录；新算子优先使用 `templates/operator/reports/` |

`templates/operator/` 是当前唯一的算子目录规范。根目录下两个 `op-*.md`
模板保留用于兼容已有报告，不再形成第二套算子目录结构；报告中的应用层必须
注明消费者类型、证据等级和引擎版本。

样例报告（真实数据）: [gelu_and_mul 测试报告](../reports/examples/gelu_and_mul_test_report.md)
· [b-fullstack 开发报告](../examples/b-fullstack/report.md)

## 自动生成报告骨架

```bash
python3 scripts/gen_report_scaffold.py --op my_op --route a2 --device p800-kunlunxin
# 产出 reports/my_op_p800-kunlunxin_report.md（环境/验证结果自动填好，
# 算子定义/实现/性能等留 TODO 手填）
```
