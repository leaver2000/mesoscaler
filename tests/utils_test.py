import itertools

import numpy as np
import pytest

import src.mesoscaler.utils as utils


def test_sort_unique() -> None:
    assert utils.sort_unique([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert utils.sort_unique([10, 12, 2]) == [2, 10, 12]
    assert utils.sort_unique([10, 12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) == [2, 10, 12]
    assert utils.sort_unique([25, 1025, 50], descending=True) == [1025, 50, 25]
    assert utils.sort_unique(itertools.chain(range(10), range(10)), descending=True) == list(range(10))[::-1]


@pytest.mark.parametrize(
    "start,stop,freq,dtype",
    [("2020-01-01T00:00:00.000000000", "2021-01-02T00:00:00.000000000", "M", "datetime64[M]")],
)
def test_date_range(start, stop, freq, dtype) -> None:
    expect = np.arange(start, stop, dtype=dtype)
    x = utils.date_range(start, stop, freq=freq)
    assert np.all(x == expect)


def test_items() -> None:
    assert (
        list(utils.items({"a": 1, "b": 2, "c": 3}))
        == list(utils.items([("a", 1), ("b", 2), ("c", 3)]))
        == list(utils.items(("a", 1), ("b", 2), ("c", 3)))
        == list(utils.items(zip("abc", (1, 2, 3))))
        == [("a", 1), ("b", 2), ("c", 3)]
    )


def test_is_pair() -> None:
    assert utils.is_pair((1, 2))
    assert not utils.is_pair((1, 2, 3))
    assert not utils.is_pair((1, 2, 3))

    assert utils.is_pair((1, "a"), strict=False)
    assert not utils.is_pair((1, "a"), strict=True)
