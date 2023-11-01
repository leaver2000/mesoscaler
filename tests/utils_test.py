from __future__ import annotations

import itertools

import pytest

import src.mesoscaler.utils as utils


def test_sort_unique() -> None:
    assert utils.sort_unique([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert utils.sort_unique([10, 12, 2]) == [2, 10, 12]
    assert utils.sort_unique([10, 12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) == [2, 10, 12]
    assert utils.sort_unique([25, 1025, 50], descending=True) == [1025, 50, 25]
    assert utils.sort_unique(itertools.chain(range(10), range(10)), descending=True) == list(range(10))[::-1]


def test_items() -> None:
    assert (
        list(utils.chain_items({"a": 1, "b": 2, "c": 3}))
        == list(utils.chain_items([("a", 1), ("b", 2), ("c", 3)]))
        == list(utils.chain_items(("a", 1), ("b", 2), ("c", 3)))
        == list(utils.chain_items(zip("abc", (1, 2, 3))))
        == [("a", 1), ("b", 2), ("c", 3)]
    )


def test_is_pair() -> None:
    assert utils.is_pair((1, 2))
    assert not utils.is_pair((1, 2, 3))
    assert not utils.is_pair((1, 2, 3))

    assert utils.is_pair((1, "a"), strict=False)
    assert not utils.is_pair((1, "a"), strict=True)


def test_pair() -> None:
    assert utils.pair(1) == (1, 1)
    assert utils.pair((1, 2)) == (1, 2)
    with pytest.raises(ValueError):
        utils.pair((1, 2, 3))


def test_triples() -> None:
    assert utils.triples(1) == (1, 1, 1)
    assert utils.triples((1, 2, 3)) == (1, 2, 3)
    with pytest.raises(ValueError):
        utils.triples((1, 2))


def test_quads() -> None:
    assert utils.quads(1) == (1, 1, 1, 1)
    assert utils.quads((1, 2, 3, 4)) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        utils.quads((1, 2, 3))


@pytest.mark.parametrize(
    "a,b",
    [
        (0, 0),
        (360, 0),
        (25, 25),
        (90, 90),
        (359, -1),
        (180, -180),
        (181, -179),
        (360, 0),
        (270, -90),
        (271, -89),
    ],
)
def test_lon_0_360_to_180_180(a, b):
    assert utils.lon_0_360_to_180_180(a) == b == utils.long3(a)


@pytest.mark.parametrize("a", range(0, 360))
def test_lon_180_180_to_0_360(a):
    b = utils.lon_0_360_to_180_180(a)
    assert utils.lon_180_180_to_0_360(b) == a == utils.long1(b)
