# gelu 黄金数据

- `inputs_spec.yaml` — 声明式规格（唯一事实来源）：模式 × dtype × shape
  × seed 全枚举 + 特殊值集 + backward 组。
- `data/` — `script/gen_golden.py --device cpu` 的生成物（可复现，
  **不入库**，.gitignore 口径与全库一致）。
- 索引: `data/index.json` 记录每条 case 的文件名/sha256/形状/dtype/模式，
  `script/check_accuracy.py --impl <级别>` 读黄金按容差判定。

生成与校验:

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device cuda:6   # 真机
python3 script/check_accuracy.py --impl native --device cuda:6   # 原生对照
python3 script/check_accuracy.py --impl torch   --device cuda:6   # torch 级对照
```

容差（docs/acceptance.md 唯一权威表）: fp32=1e-5 abs / fp16|bf16=1e-2 abs;
非有限值（inf/nan）走**位级 parity**（nan↔nan、inf↔inf），不进数值容差。
