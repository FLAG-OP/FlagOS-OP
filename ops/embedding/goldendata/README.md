# Golden data

`inputs_spec.yaml` declares forward lookup cases. Generate CPU goldens with:

```bash
python3 ../script/gen_golden.py
```

The generator cross-checks the independent index-select reference against CPU
`F.embedding`. Data and the hash index are reproducible generated artifacts and
are not committed.
