# 黄金数据

规格见 [inputs_spec.yaml](inputs_spec.yaml)（字段说明与 special 用例
见[样板 goldendata](../../../templates/operator/goldendata/README.md)）。
生成物 `data/`、`index.json` 不入库:

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device p800-kunlunxin
```
