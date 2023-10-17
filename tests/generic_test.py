import typing

from src.mesoscaler._typing import get_first_order_generic
from src.mesoscaler.generic import DataMapping, Dataset


class MyDataset(Dataset[int]):
    def __init__(self) -> None:
        super().__init__()
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __getitem__(self, index: int) -> int:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def test_dataset() -> None:
    a = MyDataset()

    b = MyDataset()
    c = a + b
    assert len(c) == 20
    assert c[0] == 1


def test_generic_data_mapping() -> None:
    data = DataMapping({"a": 1})
    # for arg in get_first_order_generic(data):
    #     assert isinstance(arg, typing.TypeVar)


class ListMapping(DataMapping[str, list[int]]):
    def __init__(self) -> None:
        super().__init__({"a": [1, 2, 3]})


def test_list_mapping() -> None:
    data = ListMapping()
    # assert get_first_order_generic(data) == (str, list[int])
