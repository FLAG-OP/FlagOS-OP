#!/usr/bin/env python3
"""Deterministically build the tiny reference model used by C2 evidence.

Identity is fully reproducible from this file: LlamaConfig(hidden=64, heads=4,
layers=2, head_dim=16), a fixed WordLevel tokenizer, torch.manual_seed(0).

Usage: python3 scripts/c2_enginecore/make_model.py /tmp/flagos_tiny_llama
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast)

WORDS = ["<unk>", "<s>", "</s>", "<pad>", "the", "future", "of", "AI",
         "is", "bright", "hello", "world", "a", "b", "c", "1", "2", "3",
         "machine", "learning", "model", "data", "test", "prompt"]
SEED = 0


def build(out: str) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>",
                            eos_token="</s>", unk_token="<unk>",
                            pad_token="<pad>").save_pretrained(str(d))
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64,
                      intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=256, bos_token_id=1,
                      eos_token_id=2, pad_token_id=3)
    torch.manual_seed(SEED)
    LlamaForCausalLM(cfg).save_pretrained(str(d))
    print(f"built {d} (seed={SEED}, vocab={len(vocab)}, hidden=64, "
          f"heads=4, layers=2, head_dim=16)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/flagos_tiny_llama")
