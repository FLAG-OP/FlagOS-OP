# Hardware level: intentionally not implemented

The current delivery has two production implementations:

- Ascend 910: Triton online-softmax forward plus mathematical backward;
- Kunlunxin P800: vendor efficient-attention forward plus semantic/precision
  shims and mathematical autograd handling.

No separate hardware-language SDPA kernel is included.

Reasons:

1. the Ascend deliverable already includes a source-level Triton kernel;
2. P800 XMLIR/Triton SDPA experiments are 5.4-7.0x slower than vendor
   efficient attention;
3. the current container does not expose the lower-level Kunlunxin kernel
   development toolchain;
4. writing another wrapper around vendor primitives would not expose the
   desired internal schedule.

Revisit this level when a Kunlunxin hardware SDK becomes available or when
XMLIR/Triton can lower the causal attention dot/loop structure near the
vendor kernel's utilization.
