from src.mesoscaler.generic import Dataset


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
