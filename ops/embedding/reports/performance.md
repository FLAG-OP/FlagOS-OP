# embedding 性能报告

日期：2026-09-23；设备：`cuda:1`；dtype：FP16。

所有 forward 计时每次读取一个输出元素，避免 XMLIR 只测 launch。

## Forward

| shape | ours | native embedding | Triton gather | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0680ms | 0.0358ms | 0.2403ms | 0.526x |
| 16k×D128 | 0.0514ms | 0.0494ms | 2.2206ms | 0.963x |
| 131k×D128 | 0.1665ms | 0.1530ms | 16.9373ms | 0.919x |
| 16k×D512 | 0.0616ms | 0.0548ms | 8.7184ms | 0.889x |
| vocab128k 16k×D128 | 0.0514ms | 0.0515ms | 2.2458ms | 1.001x |

## Backward

16k indices × D128：

| mode | ours | native |
|---|---:|---:|
| dense | 1.3306ms | 1.2993ms |
| inverse-frequency | 1.9562ms | XPU native 不支持 |

## 结论

1. 生产路径与 native embedding 在 16k 以上基本持平；
2. 1k 小 shape 有 Python wrapper / 完成守卫开销；
3. Triton gather 虽然正确，但慢约 3.5-102x，不作为生产实现；
4. inverse-frequency fallback 填补 XPU native 缺口，而不是更快路径。

复现：

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
```
