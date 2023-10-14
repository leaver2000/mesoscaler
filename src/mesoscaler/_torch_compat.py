# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false,reportMissingImports=false
# flake8: noqa: F821
"""torch optional dependency"""
from __future__ import annotations

__all__ = ["Dataset", "ConcatDataset", "IterableDataset", "ChainDataset", "Tensor"]
import abc
import bisect
import contextlib
import itertools
from typing import Generic, Iterable, Iterator, TypeVar

from numpy import ndarray as Tensor

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co], abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> T_co:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]:
        return ConcatDataset([self, other])


class ConcatDataset(Dataset[T_co]):
    @staticmethod
    def cumsum(sequence: list[Dataset[T_co]]) -> list[int]:
        lens = map(len, sequence)
        return list(itertools.accumulate(lens))

    def __init__(self, datasets: Iterable[Dataset[T_co]]) -> None:
        super().__init__()
        self.datasets = datasets = list(datasets)
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        for d in datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> T_co:
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


class IterableDataset(Dataset[T_co], abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        ...

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])


class ChainDataset(IterableDataset[T_co]):
    def __init__(self, datasets: Iterable[Dataset[T_co]]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self) -> Iterator[T_co]:
        return itertools.chain.from_iterable(self.datasets)

    def __len__(self) -> int:
        return sum(map(len, self.datasets))


with contextlib.suppress(ImportError):
    from torch import Tensor as Tensor
    from torch.utils.data import ChainDataset, ConcatDataset, Dataset, IterableDataset
