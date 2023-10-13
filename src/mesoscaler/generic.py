"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import queue
import random
import threading
from typing import Any, Callable, Generic, Iterable, TypeVar

from ._torch_compat import IterableDataset
from ._typing import (
    Any,
    AnyArrayLike,
    Callable,
    DictStrAny,
    Final,
    Generic,
    Hashable,
    HashableT,
    Iterable,
    Iterator,
    Mapping,
    NumpyDType_T,
    PandasDType_T,
    Self,
    TypeVar,
    overload,
    ListLike,
)
from .utils import is_array_like, join_kv

_T = TypeVar("_T")


_AnyArrayLikeT = TypeVar("_AnyArrayLikeT", bound=AnyArrayLike)


class Loc(Generic[_AnyArrayLikeT]):
    def __init__(
        self,
        hook: Callable[[AnyArrayLike[NumpyDType_T, PandasDType_T]], _AnyArrayLikeT],
        data: AnyArrayLike[NumpyDType_T, PandasDType_T],
    ) -> None:
        self._hook = hook
        self._data = data

    @overload
    def __getitem__(self, item: list) -> PandasDType_T | NumpyDType_T:
        ...

    @overload
    def __getitem__(self, item: Any) -> _AnyArrayLikeT:
        ...

    def __getitem__(self, item: Any) -> _AnyArrayLikeT | PandasDType_T | NumpyDType_T:  # type: ignore
        x = self._data[item]
        return self._hook(x) if is_array_like(x) else x


class Data(Generic[_T], abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> Iterable[tuple[Hashable, _T]]:
        ...

    def to_dict(self) -> dict[Hashable, _T]:
        return dict(self.data)

    def __repr__(self) -> str:
        return join_kv(self.__class__, *self.data)


class DataMapping(Mapping[HashableT, _T], Data[_T]):
    def __init__(self, data: Mapping[HashableT, _T]) -> None:
        super().__init__()
        self._data: Final[Mapping[HashableT, _T]] = data

    @property
    def data(self) -> Iterable[tuple[HashableT, _T]]:
        yield from self.items()

    def __getitem__(self, key: HashableT) -> _T:
        return self._data[key]

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class DataWorker(Mapping[HashableT, _T], Data[_T]):
    __slots__ = ("_indices", "config")

    def __init__(self, *, indices: Iterable[HashableT], **config: Any) -> None:
        super().__init__()
        self._indices: Final[list[HashableT]] = list(indices)
        self.config: Final[DictStrAny] = config

    @property
    def indices(self) -> tuple[HashableT, ...]:
        return tuple(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._indices)

    @property
    def data(self):
        x = self._indices
        return [
            ("indices", f"[{x[0]} ... {x[-1]}]"),
        ]

    def split(self, frac: float = 0.8) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self._indices[:n], self._indices[n:]
        return cls(indices=left, **self.config), cls(indices=right, **self.config)

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self._indices)
        return self

    @abc.abstractmethod
    def __getitem__(self, key: HashableT) -> _T:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...


class ABCDataConsumer(Generic[HashableT, _T], IterableDataset[_T]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, worker: DataWorker[HashableT, _T], *, maxsize: int = 0, timeout: float | None = None) -> None:
        super().__init__()
        self.thread = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue = queue.Queue[_T](maxsize=maxsize)
        self.worker = worker
        self.timeout = timeout

    def _target(self) -> None:
        for index in self.worker.keys():
            self.queue.put(self.worker[index], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[_T]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.worker.start()
        self.thread.start()
        return self
