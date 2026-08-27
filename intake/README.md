# AI 生成算子 intake 通道

把 KernelGen / KernelBench / 手写产生的 kernel 落盘成统一契约，
由 [intake_validate.py](../scripts/intake_validate.py) 自动送入三级验证。

完整说明: [AI 生成算子接入](../docs/ai-intake.md)

```
intake/
  manifest.schema.json      # 契约 JSON Schema（外部工具/Agent 可直接消费）
  cases/<case-name>/
    manifest.json           # 用例清单
    kernel.py               # 生成产物（Triton 级为 Triton，硬件级为厂商语言绑定）
    reference.py            # 参考实现（可选，未提供时用 op 语义参考）
```

验证:

```bash
python3 scripts/intake_validate.py intake/cases/kernelgen-gelu-example
python3 scripts/intake_validate.py --validate-only    # 仅契约校验（CI 用）
```
