import pytest

from src.mesoscaler.sampling.sampler import AreaOfInterestSampler


def stride_args(a=None, b=None, c=None):
    return a, b, c


@pytest.mark.parametrize(
    "args, expect",
    [
        (stride_args(1), (1, 1, 1)),
        (stride_args(1, None, 2), (1, 1, 2)),
        (stride_args((1, 2, 3)), (1, 2, 3)),
        (((1, 2), None, None), (1, 1, 2)),
    ],
)
def test_stride(args, expect):
    assert AreaOfInterestSampler._resolve_stride(*args) == expect
