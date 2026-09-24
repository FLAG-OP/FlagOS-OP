# P800 custom fixed-schedule Triton experiment

Date: 2026-09-23

## Seven-step attempt

This report records the requested attempt to build a self-managed P800 Triton
schedule rather than delegating to the vendor efficient-attention kernel.

| Step | Result |
|---|---|
| 1. P800-specific kernel | ✅ experimental forward in [../kernel/p800_custom_triton.py](../kernel/p800_custom_triton.py) |
| 2. Static causal full-block / diagonal split | ✅ `STAGE=3`: full blocks below diagonal, then diagonal stage |
| 3. Fixed schedule, no autotune | ✅ `BLOCK_M=64`, `BLOCK_N=min(64,D)`, 4 warps, 1 stage |
| 4. No-mask causal first | ✅ passes vendor comparison |
| 5. GQA / bool mask / float mask / tail | ◐ GQA, non-causal and tails pass; bool/float masks fail launch |
| 6. Force output completion while timing | ✅ `output[0,0,0,0].item()` every call |
| 7. Beat vendor efficient attention | ❌ custom is 2.1-6.5x slower in the measured cases |

The module is intentionally **not connected to `kernel.backends`** and does not
replace the production P800 vendor delegation.

## Implementation

The experimental kernel is derived from FlagGems 4.2.1rc0's Apache-2.0
`_kunlunxin` attention forward, but reduced to:

- one deterministic forward kernel;
- no Python autotuner;
- no backward path;
- no flash API wrapper;
- no output log-sumexp allocation/store;
- fixed `BLOCK_M/BLOCK_N/warps/stages`;
- explicit causal full-block and diagonal stages;
- a public wrapper restricted to the stable no-mask path.

The mask branch remains in the private launch function for diagnostic probes,
but the public experimental wrapper rejects masks.

## Correctness

Reproducer:

```bash
python3 ops/sdpa/script/probe_p800_custom_schedule.py \
  --device cuda:1 \
  --json-out ops/sdpa/reports/p800-custom-schedule.json
```

No-mask results:

| Case | Max error vs vendor |
|---|---:|
| B1 H4/4 S128 D128 causal | 1.9e-6 |
| B1 H32/8 S1024 D128 causal GQA | 3.8e-6 |
| B1 H4/4 S333 D64 causal tail | 0 |
| B1 H8/2 S257 D64 non-causal GQA | 0 |
| B1 H4/4 Sq1/Skv7 D64 decode shape | 0 |

Thus the restricted schedule handles:

- causal and non-causal;
- GQA;
- sequence tails;
- `Sq != Skv` in non-causal mode.

Mask probes still fail for both bool and float additive bias:

```text
xpuLaunchKernel("_attn_fwd", ...) -> Operation not permitted(err_code: 1)
```

Raw data: [p800-custom-schedule.json](p800-custom-schedule.json)

## Performance

All timed calls consume an output element on the host.

| Case | Vendor efficient | Custom fixed schedule | Custom slowdown |
|---|---:|---:|---:|
| B1 H16 S1k D64 | 0.1503 ms / 14.3 TF | 0.3206 ms / 6.7 TF | **2.13x** |
| B1 H16 S1k D128 | 0.1566 ms / 27.4 TF | 0.3354 ms / 12.8 TF | **2.14x** |
| B1 H16 S4k D128 | 0.6706 ms / 102.5 TF | 4.3509 ms / 15.8 TF | **6.49x** |
| B16 H16 S1k D128 | 0.8007 ms / 85.8 TF | 4.4334 ms / 15.5 TF | **5.54x** |

The seventh goal is therefore not met. In the FA2-style D128 S1k case, the
custom schedule is also about **13x slower than the public A100·FA2 reference**
derived earlier from paper throughput.

## Schedule variants tried

The following changes did not remove the long-sequence performance plateau:

- diagonal-first versus full-blocks-first;
- `tl.exp` versus `tl.math.exp2`;
- folding scale into Q before launch;
- fp32 dot inputs;
- `BLOCK_M/BLOCK_N` changes;
- `num_warps` changes;
- `num_stages=1..4`;
- preloading V;
- removing `allow_tf32=False`;
- removing mask/LSE work from the forward-only specialization.

A raw-pointer rewrite failed XMLIR rewriting:

```text
Rewrite for-op failed. Could not find PtrState returned by the loop.
```

The custom module therefore retains the upstream-style pointer structure that
compiles on this backend.

## Conclusion

The experiment establishes a useful restricted kernel and reproducer, but it
also closes the current scheduling hypothesis:

1. A deterministic Triton schedule can be made correct for no-mask causal,
    GQA and tails.
2. Existing mask lowering remains broken.
3. Even the stripped fixed schedule reaches only about **6-16 TFLOPS** in the
    longer/larger cases.
4. Vendor efficient attention reaches roughly **86-103 TFLOPS** in comparable
    cases.

The next credible route is not more Python-side tile tuning; it requires either:

- a new XMLIR/Triton lowering path for the dot/loop structure; or
- access to lower-level Kunlunxin kernel development tools; or
- acceptance of the vendor efficient kernel as the production P800 path.
