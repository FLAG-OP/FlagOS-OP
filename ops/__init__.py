"""ops — FlagOS-OP 已交付算子的包命名空间。

包化（多算子同进程冲突的根治, 见 sdpa reports/e2e_mini_llm.md）:
  from ops.sdpa import register_a1          # 顶层名带包前缀
  from ops.embedding import register_a1     # 与上者共存无冲突

各算子目录仍可直接 sys.path 注入独立运行（单算子脚本模式不变）。
不做任何 eager import（避免拉起 torch/triton）。
"""
