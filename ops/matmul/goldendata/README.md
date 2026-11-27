# matmul 黄金数据

- `inputs_spec.yaml` — 声明式规格（唯一事实来源）。
- `data/` — `script/gen_golden.py --device cpu` 生成物（可复现，不入库）。
- 索引: `data/index.json`（文件名/sha256/形状/dtype/参数）。

生成与校验:

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl native --device cuda:6   # 真机
python3 script/check_accuracy.py --impl torch   --device cuda:6  # torch 级
```

容差（docs/acceptance.md 唯一权威表）: GEMM 形态 5e-2 rel。
