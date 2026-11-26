"""CPU semantic oracle and validation; does not claim accelerator coverage."""
import pytest
import torch

from ops.scatter_.test.cases import cases
from ops.scatter_.validation import validate


def test_valid_metadata_matches_native():
    for name, dst, dim, index, src in cases():
        validate(dst, dim, index, src)
        assert dst.scatter_(dim, index, src) is dst, name


@pytest.mark.parametrize("kind", ["dim", "index_dtype", "src_dtype", "rank", "size"])
def test_invalid_metadata_matches_native(kind):
    dst, dim = torch.zeros(3, 5), 1
    index, src = torch.zeros(3, 5, dtype=torch.long), torch.ones(3, 5)
    if kind == "dim":
        dim = 2
    elif kind == "index_dtype":
        index = index.float()
    elif kind == "src_dtype":
        src = src.double()
    elif kind == "rank":
        index = index.flatten()
    elif kind == "size":
        index = torch.zeros(4, 5, dtype=torch.long)
    with pytest.raises((RuntimeError, IndexError)):
        dst.scatter_(dim, index, src)
    with pytest.raises((RuntimeError, IndexError)):
        validate(dst, dim, index, src)


def test_reject_overlap_and_alias():
    with pytest.raises(NotImplementedError, match="overlapping"):
        validate(torch.zeros(1).expand(5), 0, torch.arange(5), 1.)
    dst = torch.zeros(5)
    with pytest.raises(NotImplementedError, match="aliased"):
        validate(dst, 0, torch.arange(5), dst)
