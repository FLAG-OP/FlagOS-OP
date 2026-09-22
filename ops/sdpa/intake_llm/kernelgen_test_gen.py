import pytest
import triton
from scaled_dot_product_attention_triton import scaled_dot_product_attention as scaled_dot_product_attention_triton
from scaled_dot_product_attention_torch import scaled_dot_product_attention as scaled_dot_product_attention_baseline

import torch

@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("shape", [(2, 4, 64, 32), (1, 8, 128, 64), (2, 16, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scaled_dot_product_attention(shape, dtype):
    batch_size, num_heads, seq_len, head_dim = shape
    query = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    key = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    value = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device='cuda')

    ref_query = query
    ref_key = key
    ref_value = value

    ref_out = scaled_dot_product_attention_baseline(ref_query, ref_key, ref_value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
    res_out = scaled_dot_product_attention_triton(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)

    torch.testing.assert_close(res_out, ref_out, rtol=1e-3, atol=1e-3)