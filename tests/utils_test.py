import itertools

from src.mesoscaler.utils import sort_unique


def test_sort_unique() -> None:
    assert sort_unique([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sort_unique([10, 12, 2]) == [2, 10, 12]
    assert sort_unique([10, 12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) == [2, 10, 12]
    assert sort_unique([25, 1025, 50], descending=True) == [1025, 50, 25]
    assert sort_unique(itertools.chain(range(10), range(10)), descending=True) == list(range(10))[::-1]
