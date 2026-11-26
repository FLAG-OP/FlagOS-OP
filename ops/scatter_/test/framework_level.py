"""C1 mini consumer: construct a token-selection mask. Not real vLLM C2."""
import torch

from ops.scatter_ import register_a1


def run(profile):
    scores = torch.arange(3 * 1031, device=profile.torch_device).float().reshape(3, 1031)
    tokens = torch.tensor([[0, 17, 1030], [2, 53, 999], [1, 7, 101]], device=scores.device)

    def consumer():
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, tokens, True)
        return scores.masked_fill(~mask, float("-inf")).softmax(-1)

    expected = consumer()
    counter = {"n": 0}
    lib = register_a1("CUDA", counter=counter, platform=profile.name)
    try:
        with torch.no_grad():
            actual = consumer()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert counter["n"] == 1
        return {"ok": True, "hits": 1, "consumer": "token-selection-mini"}
    finally:
        lib._destroy()
