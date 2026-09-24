# P800 direct Triton SDPA probe

Date: 2026-09-23

## Question

Kunlunxin has a Triton/XPU backend and FlagGems ships a `_kunlunxin` SDPA
kernel. This probe answers a narrower engineering question: if the current
vendor efficient-attention delegation is replaced by the existing Triton path,
what accuracy and performance do we get today?

The locked image does **not** contain a separate Python package named
`gtriton`. The available source-level stack is:

- Triton `3.0.0+03c4c9be`
- `torch_xmlir` `XMLIR--bc1b1dc6f-dev+2026032411`
- FlagGems `4.2.1rc0`

## Method

Reproducer: [../script/probe_p800_triton.py](../script/probe_p800_triton.py)

- Directly call FlagGems `_kunlunxin.ops.attention._attn_fwd.fn`, bypassing
  Python autotune.
- Use deterministic `BLOCK_M/BLOCK_N/num_warps/num_stages` configurations.
- Restrict the first probe to the stable no-mask causal fp16 path.
- Compare against the current P800 backend, which directly delegates to
  `aten::_scaled_dot_product_efficient_attention`.
- Consume one output element through `.item()` in every timed call. This is
  required on XMLIR; discarding the output can measure only launch/submission.
- Check the direct Triton output against the vendor path.

Raw data:

- Microbenchmark: [p800-triton-micro.json](p800-triton-micro.json)
- FA2-style 16k-token protocol: [p800-triton-fa2.json](p800-triton-fa2.json)

## Accuracy

For no-mask causal fp16 cases, the direct Triton output is numerically sound:

| Case | max error vs vendor |
|---|---:|
| B1 H16 S1024 D64 | 1.9e-6 |
| B1 H16 S1024 D128 | 7.6e-6 |
| B1 H16 S4096 D128 | 1.9e-6 |
| B16 H32 S1024 D64 | 7.6e-6 |
| B16 H16 S1024 D128 | 3.8e-6 |
| B4 H16 S4096 D128 | 3.8e-6 |

This is a restricted-path accuracy result. It does not override the earlier
finding that a float mask can fail launch (`xpuLaunchKernel ... Operation not
permitted`).

## B=1 microbenchmark

| Case | vendor efficient | best direct Triton | direct Triton slowdown |
|---|---:|---:|---:|
| H16 S1k D64 | 0.1426 ms | 0.2768 ms | **1.94x** |
| H16 S1k D128 | 0.1569 ms | 0.3111 ms | **1.98x** |
| H16 S4k D128 | 0.6706 ms | 4.2961 ms | **6.41x** |

## FA2-style total-token protocol

The protocol keeps `B×S=16384` tokens constant and uses causal fp16.

| Case | vendor efficient | best direct Triton | direct Triton slowdown |
|---|---:|---:|---:|
| D64, B16 H32 S1k | 1.1961 ms / 57.4 TF | 7.6075 ms / 9.0 TF | **6.36x** |
| D128, B16 H16 S1k | 0.7946 ms / 86.4 TF | 4.3294 ms / 15.9 TF | **5.45x** |
| D128, B4 H16 S4k | 2.3123 ms / 118.8 TF | 16.0643 ms / 17.1 TF | **6.95x** |

For reference, the vendor backend itself is about 1.9-2.8x slower than public
A100 FlashAttention-2 paper data under the same protocol. Therefore this
existing Triton path is roughly an order of magnitude slower than A100·FA2 in
the D128 cases; it is not a competitive production replacement.

## Sensitivity and failed rewrite

- Changing `BLOCK_M/BLOCK_N` among the tested values changed latency by only
  a few percent.
- Changing `num_stages` from 1 through 4 was effectively neutral.
- Changing `num_warps` and preloading V were also not material.
- Removing `allow_tf32=False` from the two dots did not improve performance.
- A minimal raw-pointer rewrite failed in XMLIR Triton rewriting with:

```text
error: Rewrite for-op failed. Could not find PtrState returned by the loop.
```

This is the same class of frontend/rewrite obstacle encountered when directly
porting the Ascend kernel. The existing FlagGems kernel works because it uses
its specific block-pointer structure, but that structure is currently mapped
poorly enough to create a multi-millisecond plateau at longer sequences.

## Conclusion

“Kunlunxin Triton backend exists” and “existing SDPA kernel is production
competitive” are different claims. On the locked image:

1. The existing Triton path is numerically usable for no-mask causal fp16.
2. It is **5.4-7.0x slower** than vendor efficient attention under the FA2-style
   protocol.
3. Its mask coverage is incomplete.
4. A source-level custom schedule is possible in principle, but a simple
   raw-pointer rewrite hits XMLIR frontend limits.

The practical next step for a self-managed schedule is a P800-specific kernel
specialization project (static causal block split, stable tail handling, then
mask/GQA), not simply switching the default backend to the existing FlagGems
kernel. Until that project beats the vendor path, the vendor delegation remains
the safer production route.
