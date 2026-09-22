import pytest
from scaled_dot_product_attention_triton import scaled_dot_product_attention as scaled_dot_product_attention_triton
from scaled_dot_product_attention_torch import scaled_dot_product_attention as scaled_dot_product_attention_baseline

import torch

@pytest.mark.scaled_dot_product_attention_benchmark
@pytest.mark.parametrize("shape", [(2, 4, 64, 32), (1, 8, 128, 64), (2, 16, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def scaled_dot_product_attention_benchmark(shape, dtype):
    import torch.utils.benchmark as benchmark
    import triton

    quantiles = [0.5, 0.2, 0.8]

    batch_size, num_heads, seq_len, head_dim = shape
    query = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    key = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    value = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')

    ref_query = query
    ref_key = key
    ref_value = value

    ms_torch, _, _ = triton.testing.do_bench(lambda: scaled_dot_product_attention_baseline(ref_query, ref_key, ref_value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None), rep=100, quantiles=quantiles)
    ms_triton, _, _ = triton.testing.do_bench(lambda: scaled_dot_product_attention_triton(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None), rep=100, quantiles=quantiles)

    speedup = ms_torch / ms_triton
    print(f"PyTorch: {ms_torch:.4f}ms | Triton: {ms_triton:.4f}ms | Speedup: {speedup:.2f}x")
