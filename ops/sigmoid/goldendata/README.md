# sigmoid 黄金数据

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

仓库只提交输入规格；`data/` 与 `index.json` 由脚本生成并已在
`.gitignore` 中排除。

```bash
python ops/sigmoid/script/gen_golden.py --device cpu
python ops/sigmoid/script/check_accuracy.py \
  --impl triton --device p800-liyanz-local
```

规格覆盖 fp16、bf16、fp32，小尺寸、启动阈值前后、二维长尾、典型
Transformer 三维/四维激活，以及 `-100/-20/0/20/100` 一类饱和值。
重点边界包括 `255/256/257`、`511/512/513` 和
`65535/65536/65537`。参考输出来自
`reference.py`，低精度输入统一在 fp32 中计算后转换回输入 dtype。
