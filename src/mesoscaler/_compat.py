# noqa
# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false
"""torch==2.1.0 compatibility layer."""
from __future__ import annotations

__all__ = [
    "Dataset",
    "IterableDataset",
    "ConcatDataset",
    "ChainDataset",
]
try:
    from torch.utils.data import ChainDataset, ConcatDataset, Dataset, IterableDataset
except ImportError:
    import bisect
    import warnings
    from typing import Generic, Iterable, Iterator, List, TypeVar

    T_co = TypeVar("T_co", covariant=True)

    class Dataset(Generic[T_co]):
        def __getitem__(self, index) -> T_co:
            raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

        def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
            return ConcatDataset([self, other])

    class IterableDataset(Dataset[T_co]):
        def __iter__(self) -> Iterator[T_co]:
            raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

        def __add__(self, other: Dataset[T_co]):
            return ChainDataset([self, other])

    class ConcatDataset(Dataset[T_co]):
        datasets: List[Dataset[T_co]]
        cumulative_sizes: List[int]

        @staticmethod
        def cumsum(sequence):
            r, s = [], 0
            for e in sequence:
                l = len(e)
                r.append(l + s)
                s += l
            return r

        def __init__(self, datasets: Iterable[Dataset]) -> None:
            super().__init__()
            self.datasets = list(datasets)
            assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
            for d in self.datasets:
                assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
            self.cumulative_sizes = self.cumsum(self.datasets)

        def __len__(self):
            return self.cumulative_sizes[-1]

        def __getitem__(self, idx):
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]

        @property
        def cummulative_sizes(self):
            warnings.warn(
                "cummulative_sizes attribute is renamed to " "cumulative_sizes", DeprecationWarning, stacklevel=2
            )
            return self.cumulative_sizes

    class ChainDataset(IterableDataset):
        def __init__(self, datasets: Iterable[Dataset]) -> None:
            super().__init__()
            self.datasets = datasets

        def __iter__(self):
            for d in self.datasets:
                assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
                yield from d

        def __len__(self):
            total = 0
            for d in self.datasets:
                assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
                total += len(d)  # type: ignore[arg-type]
            return total
