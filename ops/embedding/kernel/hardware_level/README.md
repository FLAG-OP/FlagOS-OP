# Hardware level: intentionally not implemented

This delivery uses the XMLIR/P800 native row-gather (`aten::index_select`) as
the production forward implementation and native dense
`aten::embedding_backward` as the dense backward implementation.

A separate Kunlunxin hardware-language implementation is intentionally left
empty because:

1. the native gather already reaches parity with `aten::embedding`;
2. the local flattened Triton gather is 3.5-102x slower, and the P800
   specialization of FLAG-OP/gatherFIX is 10.9-177.9x slower;
3. writing another source-level gather would duplicate the vendor kernel
   without exposing new scheduling freedom for this memory-bound operation.

Revisit this level only if a future requirement needs fused embedding +
optimizer updates, cache-aware sparse lookup, or another layout-specific
kernel that the ATen primitive cannot express.
