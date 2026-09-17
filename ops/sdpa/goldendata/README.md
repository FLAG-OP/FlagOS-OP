# 黄金数据（goldendata）

- 规格: [inputs_spec.yaml](inputs_spec.yaml)（声明式，KernelBench 风格）
- 生成: `python3 script/gen_golden.py`（CPU fp32 权威，265 组）
- 判定: `python3 script/check_accuracy.py --impl <triton|native|reference> --device <dev>`
- `data/` 与 `index.json` 为生成物（sha256 索引），可随时重建

## 与模板的差异说明

- 输入维度按 SDPA 语义改为 B/Hq/Hkv/S/D + causal/mask 组合（模板为
  单一 shape 列表）
- 生成时**双向互验**: expected 同时用本目录参考实现与 CPU 官方
  F.scaled_dot_product_attention(fp32) 计算，分歧超阈值即中断——
  阈值按 dtype 分档（fp32 2e-5 / fp16 2e-2），extreme 用相对 2e-3
  （依据见[精度分册](../reports/accuracy.md)§3）
- extreme（scale=8 饱和）case 仅配 fp32: fp16/bf16 输入量化会翻转
  饱和 softmax 的 argmax（dtype 固有，实测原生同误差 1.39e-1）
