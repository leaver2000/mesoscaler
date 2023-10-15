import typing

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
    a.__first_order_generics__ = (int,)
    b = MyDataset()
    c = a + b
    assert len(c) == 20
    assert c[0] == 1


def test_generic_data_mapping() -> None:
    data = DataMapping({"a": 1})
    for x in data.__first_order_generics__:
        assert isinstance(x, typing.TypeVar)


class ListMapping(DataMapping[str, list[int]]):
    def __init__(self) -> None:
        super().__init__({"a": [1, 2, 3]})


def test_list_mapping() -> None:
    data = ListMapping()
    assert data.__first_order_generics__ == (str, list[int])
