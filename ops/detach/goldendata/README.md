# 黄金数据目录

**入库**: 只有 `inputs_spec.yaml`（规格）。
**生成物**（`data/`、`index.json`）体积大，随用随生成，不入库:

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
```

## 规格与判定

本算子的黄金规格以 `inputs_spec.yaml` 为唯一事实来源。该算子不使用
逐元素数值模板：分配/别名/dtype 元数据算子比较 shape、dtype、stride、device、
memory_format 或别名/autograd 属性；host 标量算子比较标量值、Python 类型和错误路径。
具体覆盖数量与判定口径见对应的 `reports/accuracy.md`。
