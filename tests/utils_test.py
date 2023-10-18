import itertools
import pytest
from src.mesoscaler.utils import sort_unique, date_range
import numpy as np


def test_sort_unique() -> None:
    assert sort_unique([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sort_unique([10, 12, 2]) == [2, 10, 12]
    assert sort_unique([10, 12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) == [2, 10, 12]
    assert sort_unique([25, 1025, 50], descending=True) == [1025, 50, 25]
    assert sort_unique(itertools.chain(range(10), range(10)), descending=True) == list(range(10))[::-1]


@pytest.mark.parametrize(
    "start,stop,freq,dtype",
    [("2020-01-01T00:00:00.000000000", "2021-01-02T00:00:00.000000000", "M", "datetime64[M]")],
)
def test_date_range(start, stop, freq, dtype) -> None:
    expect = np.arange(start, stop, dtype=dtype)
    x = date_range(start, stop, freq=freq)
    assert np.all(x == expect)
