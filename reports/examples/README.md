# 示例报告

| 报告 | 类型 | 状态 |
|---|---|---|
| [sdpa_mlu590_test_report.md](sdpa_mlu590_test_report.md) | **测试报告**（模板 [op-test-report](../../templates/op-test-report.md)，MLU590 三平台 SDPA） | 定稿 |
| [sdpa_mlu590_report.md](sdpa_mlu590_report.md) | 开发报告（模板 [op-development-report](../../templates/op-development-report.md)） | 定稿 |
| [gelu_and_mul_test_report.md](gelu_and_mul_test_report.md) | **测试报告**（模板 [op-test-report](../../templates/op-test-report.md) 的已填样例，真实数据） | 定稿 |
| [gelu_and_mul_p800-kunlunxin_report.md](gelu_and_mul_p800-kunlunxin_report.md) | 开发报告 | 脚手架骨架（env 自动填，正文 TODO） |
| [silu_and_mul_p800-kunlunxin_report.md](silu_and_mul_p800-kunlunxin_report.md) | 开发报告 | 脚手架骨架 |
| [b-fullstack 开发报告](../../examples/b-fullstack/report.md) | 开发报告 | 定稿（完整手写范本） |
| [softmax 开发报告](../../examples/softmax-fullstack/reports/development.md) | 开发报告 | 定稿（样板同构样例，7 章全填） |
| [softmax 测试报告](../../examples/softmax-fullstack/reports/test-report.md) | 测试报告 | 定稿（范围矩阵 + 三方精度/性能） |

新建报告: 复制 [templates/](../../templates/) 下对应模板，或
`python3 scripts/gen_report_scaffold.py --op <算子> --route <路线> --device <profile>`
生成半自动骨架（产出在 `reports/`，不入库）。
