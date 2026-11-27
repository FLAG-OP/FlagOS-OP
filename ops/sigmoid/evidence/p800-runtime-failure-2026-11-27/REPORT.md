# aten.sigmoid P800 运行时分阶段故障材料

> 该材料仅用于环境诊断，不构成 C1/C2 或性能验收结论。

| 项 | 值 |
|---|---|
| 采集时间 | `2026-11-27T15:08:18.137267+00:00` |
| Git SHA | `cb62c806476bf783a49667c32b2c764d9a254b6b` |
| profile | `p800-kunlunxin-gems5` |
| 可见设备 | `{"CUDA_VISIBLE_DEVICES": null, "XPU_VISIBLE_DEVICES": null}` |
| 容器镜像 | `e180a0dd4267` |
| 诊断结果 | **FAIL** |
| 首个失败阶段 | `generic_triton_257` |
| 通过 / 失败 / 超时 | 9 / 2 / 0 |

## 分阶段结果

| 阶段 | 状态 | 退出码 | 秒 | stdout / stderr |
|---|---:|---:|---:|---|
| `identity` | PASS | 0 | 2.262 | [identity.stdout.log](identity.stdout.log) / [identity.stderr.log](identity.stderr.log) |
| `copy_17` | PASS | 0 | 2.233 | [copy_17.stdout.log](copy_17.stdout.log) / [copy_17.stderr.log](copy_17.stderr.log) |
| `copy_255` | PASS | 0 | 2.23 | [copy_255.stdout.log](copy_255.stdout.log) / [copy_255.stderr.log](copy_255.stderr.log) |
| `copy_14336` | PASS | 0 | 2.245 | [copy_14336.stdout.log](copy_14336.stdout.log) / [copy_14336.stderr.log](copy_14336.stderr.log) |
| `native_sigmoid_17` | PASS | 0 | 2.223 | [native_sigmoid_17.stdout.log](native_sigmoid_17.stdout.log) / [native_sigmoid_17.stderr.log](native_sigmoid_17.stderr.log) |
| `native_sigmoid_255` | PASS | 0 | 2.245 | [native_sigmoid_255.stdout.log](native_sigmoid_255.stdout.log) / [native_sigmoid_255.stderr.log](native_sigmoid_255.stderr.log) |
| `native_sigmoid_14336` | PASS | 0 | 2.228 | [native_sigmoid_14336.stdout.log](native_sigmoid_14336.stdout.log) / [native_sigmoid_14336.stderr.log](native_sigmoid_14336.stderr.log) |
| `generic_triton_257` | FAIL | 1 | 5.996 | [generic_triton_257.stdout.log](generic_triton_257.stdout.log) / [generic_triton_257.stderr.log](generic_triton_257.stderr.log) |
| `sigmoid_257` | PASS | 0 | 10.052 | [sigmoid_257.stdout.log](sigmoid_257.stdout.log) / [sigmoid_257.stderr.log](sigmoid_257.stderr.log) |
| `sigmoid_255` | PASS | 0 | 10.102 | [sigmoid_255.stdout.log](sigmoid_255.stdout.log) / [sigmoid_255.stderr.log](sigmoid_255.stderr.log) |
| `vllm_tp2_c2` | FAIL | 1 | 47.946 | [vllm_tp2_c2.stdout.log](vllm_tp2_c2.stdout.log) / [vllm_tp2_c2.stderr.log](vllm_tp2_c2.stderr.log) |

## 身份与完整性

- profile: `configs/devices/p800-kunlunxin-gems5.yaml`，SHA256 `9a4c17a48b2303b012387dafee891759982316597bad875b800288aa32c53a1b`
- env lock: `configs/env/p800-kunlunxin-gems5.lock.yaml`，SHA256 `1e9837d2c73f1432b49431892b0982213206d4ad3983134d96b73d43b3d66d63`
- manifest: [manifest.json](manifest.json)
- 设备快照、进程快照和每个日志的 SHA256 均在 manifest 中。
