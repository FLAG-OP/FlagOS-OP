# 黄金数据目录

**入库**: 只有 `inputs_spec.yaml`（规格）。
**生成物**（`data/`、`index.json`）体积大，随用随生成，不入库:

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
```

## 规格字段（inputs_spec.yaml）

| 字段 | 含义 | 示例 |
|---|---|---|
| `shapes` | 多组维度（覆盖非 BLOCK 整倍数，防 [#15](../../../docs/known-issues.md) 类尾块坑） | `[[1,14336],[8,5000]]` |
| `dtypes` | 精度 | bf16/fp16/fp32 |
| `distribution` | 随机分布 | normal/uniform |
| `scale` | 分布幅度 | 2.0 |
| `seeds_per_case` | 每组合的随机种子数（数量维度） | 3 |
| `special` | 特殊功能用例 | 见下表 |

## special 用例

| 值 | 注入内容 | 抓什么问题 |
|---|---|---|
| `zeros` | 全零行/块 | 除零、0 的处理分支 |
| `large` | ±1e4 量级行 | exp 溢出、精度坍塌 |
| `boundary` | dtype 上下限附近的值 | 饱和/截断处理 |

黄金参考 = `../reference.py` 的 fp32 实现（cast 回输入 dtype），
与 `scripts/accuracy_report.py` 同口径。
