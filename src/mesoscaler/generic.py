"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import bisect
import itertools
import queue
import random
import threading
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sized,
    TypeVar,
    overload,
)

from ._typing import (
    AnyArrayLike,
    DictStrAny,
    HashableT,
    NumpyDType_T,
    PandasDType_T,
    Self,
)
from .utils import acc_size, is_array_like, join_kv

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


class NamedAndSized(Sized, abc.ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def size(self) -> int:
        return len(self)


# =====================================================================================================================
#
# =====================================================================================================================
class Data(NamedAndSized, Generic[_T], abc.ABC):
    """
    ```
    >>> class MyData(Data[int]):
    ...     def __init__(self):
    ...         self.node_1 = 0
    ...         self.node_2 = 0
    ...     @property
    ...     def nodes(self):
    ...         return "node_1", "node_2"
    ...     @property
    ...     def data(self):
    ...         return ((node, getattr(self, node)) for node in self.nodes)
    ...     def __len__(self):
    ...         return len(self.nodes)
    ...
    >>> data = MyData()
    >>> data
    MyData:
    - node_1: 0
    - node_2: 0
    >>> data.to_dict()
    {'node_1': 0, 'node_2': 0}
    ```
    """

    @property
    @abc.abstractmethod
    def data(self) -> Iterable[tuple[Hashable, _T]]:
        ...

    def __repr__(self) -> str:
        return join_kv(self.__class__, *self.data)

    def to_dict(self) -> dict[Hashable, _T]:
        return dict(self.data)


class MappingBase(Data[_T], Mapping[HashableT, _T], abc.ABC):
    @property
    def data(self) -> Iterable[tuple[Hashable, _T]]:
        return self.items()


# =====================================================================================================================
#
# =====================================================================================================================
class Dataset(NamedAndSized, Generic[_T_co], abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> _T_co:
        ...

    def __add__(self, other: Dataset[_T_co]) -> ConcatDataset[_T_co]:
        return ConcatDataset([self, other])


class ConcatDataset(Dataset[_T_co]):
    def __init__(self, data: Iterable[Dataset[_T_co]]) -> None:
        super().__init__()
        self.data = data = list(data)
        if not data:
            raise ValueError("datasets should not be an empty iterable")
        for ds in data:
            if isinstance(ds, IterableDataset):
                raise ValueError("ConcatDataset does not support IterableDataset")

        self.accumulated_sizes = list(acc_size(data))

    def __len__(self) -> int:
        return self.accumulated_sizes[-1]

    def __getitem__(self, idx: int) -> _T_co:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)

        if dataset_idx := bisect.bisect_right(self.accumulated_sizes, idx):
            idx -= self.accumulated_sizes[dataset_idx - 1]

        return self.data[dataset_idx][idx]


# =====================================================================================================================
#
# =====================================================================================================================
class IterableDataset(NamedAndSized, Generic[_T_co], abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[_T_co]:
        ...

    def __add__(self, other: IterableDataset[_T_co]) -> ChainDataset[_T_co]:
        return ChainDataset([self, other])


class ChainDataset(IterableDataset[_T_co]):
    def __init__(self, datasets: Iterable[IterableDataset[_T_co]]) -> None:
        super().__init__()
        self.data = datasets

    def __iter__(self) -> Iterator[_T_co]:
        return itertools.chain.from_iterable(self.data)

    def __len__(self) -> int:
        return sum(map(len, self.data))


# =====================================================================================================================
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
    def __getitem__(self, item: list) -> PandasDType_T | NumpyDType_T:  # pyright: ignore
        ...

    @overload
    def __getitem__(self, item: Any) -> _AnyArrayLikeT:
        ...

    def __getitem__(self, item: Any) -> _AnyArrayLikeT | PandasDType_T | NumpyDType_T:  # type: ignore
        x = self._data[item]
        return self._hook(x) if is_array_like(x) else x


class DataMapping(MappingBase[HashableT, _T]):
    def __init__(self, data: Mapping[HashableT, _T]) -> None:
        super().__init__()
        self._data = dict(data)

    def __getitem__(self, key: HashableT) -> _T:
        return self._data[key]

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return join_kv(self.__class__, *self.items())


class DataMutableMapping(MutableMapping[HashableT, _T], DataMapping[HashableT, _T]):
    def __setitem__(self, key: HashableT, value: _T) -> None:
        self._data[key] = value

    def __delitem__(self, key: HashableT) -> None:
        del self._data[key]


# =====================================================================================================================
class DataWorker(MappingBase[HashableT, _T]):
    __slots__ = ("indices", "config")

    def __init__(self, *, indices: Iterable[HashableT], **config: Any) -> None:
        super().__init__()
        self.indices: Final[list[HashableT]] = list(indices)
        self.config: Final[DictStrAny] = config

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self.indices)

    @abc.abstractmethod
    def __getitem__(self, key: HashableT) -> _T:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...

    def split(self, frac: float = 0.8) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self.indices[:n], self.indices[n:]
        return cls(indices=left, **self.config), cls(indices=right, **self.config)

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self.indices)
        return self


class DataConsumer(IterableDataset[_T], Generic[HashableT, _T]):
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
