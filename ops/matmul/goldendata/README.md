# Matmul golden data

`inputs_spec.yaml` declares 16 paired input cases, three dtypes and three seeds
per case (144 groups). The final `rank_one` case covers `K=1` without changing
the seeds of the earlier 135 groups. `script/gen_golden.py` computes forward and both input gradients on
CPU, then writes `data/*.pt` plus a SHA-256 `index.json`. Generated files are
ignored by Git. Each filename contains a unique case/dtype/seed combination;
the generator rejects duplicate filenames.

Run from `ops/matmul/`:

```bash
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl torch --device cpu
python3 script/check_accuracy.py --impl triton --device <profile-name>
```

Forward and gradients use the repository's GEMM relative tolerance. Cases
with an all-zero reference additionally require a zero output within the
selected dtype's absolute tolerance.
