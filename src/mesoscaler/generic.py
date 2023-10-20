"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

__all__ = [
    "NamedAndSized",
    "Data",
    "DataMapping",
    "DataWorker",
    "DataGenerator",
    # - torch.utils.data
    "Dataset",
    "IterableDataset",
    "ConcatDataset",
    "ChainDataset",
    "BatchSampler",
    "Sampler",
    "SequentialSampler",
]
import abc
import itertools
import queue
import random
import threading

from ._compat import (  # noqa
    BatchSampler,
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    Sampler,
    SequentialSampler,
)
from ._typing import (
    Any,
    Callable,
    Final,
    Generic,
    Hashable,
    HashableT,
    Iterable,
    Iterator,
    Mapping,
    Self,
    Sequence,
    Sized,
    TypeVar,
    get_first_order_generic,
    overload,
)
from .utils import join_kv, repr_

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


class NamedAndSized(Sized, abc.ABC):
    __slots__ = ()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def size(self) -> int:
        return len(self)


# =====================================================================================================================
#
# =====================================================================================================================
class Data(NamedAndSized, Generic[_T1], abc.ABC):
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
    def data(self) -> Iterable[tuple[Hashable, _T1]]:
        ...

    def __repr__(self) -> str:
        name = self.name
        data = self.data
        size = self.size
        text = join_kv(f"{name}({size=}):", *data)
        return text

    def to_dict(self) -> dict[Hashable, _T1]:
        return dict(self.data)


# =====================================================================================================================
# - Iterables
# =====================================================================================================================
class DataSampler(NamedAndSized, Iterable[_T1], Sampler[_T1], abc.ABC):
    ...


class DataSequence(NamedAndSized, Sequence[_T1]):
    def __init__(self, data: Iterable[_T1]) -> None:
        super().__init__()
        self._data = data

    @property
    def data(self) -> tuple[_T1, ...]:
        if not isinstance(self._data, tuple):
            self._data = tuple(self._data)
        return self._data

    def __iter__(self) -> Iterator[_T1]:
        return iter(self._data)

    @overload
    def __getitem__(self, idx: int, /) -> _T1:
        ...

    @overload
    def __getitem__(self, idx: slice, /) -> Self:
        ...

    def __getitem__(self, idx: int | slice) -> _T1 | Self:
        if isinstance(idx, slice):
            return self.__class__(self.data[idx])
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        name = self.name
        size = self.size
        text = "\n".join(repr_(x) for x in self.data)
        return f"{name}({size=})[\n{text}\n]"

    @overload
    def map(self, func: Callable[[_T1], _T1]) -> Self:
        ...

    @overload
    def map(self, func: Callable[[_T1], _T2]) -> DataSequence[_T2]:
        ...

    def map(self, func: Callable[[_T1], _T1] | Callable[[_T1], _T2]) -> Self | DataSequence[_T2]:
        return self.__class__(map(func, self.data))


# =====================================================================================================================
# - Mappings
# =====================================================================================================================
class DataMapping(NamedAndSized, Mapping[HashableT, _T1]):
    def __init__(self, data: Mapping[HashableT, _T1]) -> None:
        super().__init__()
        self._data = dict(data)

    def __getitem__(self, key: HashableT) -> _T1:
        return self._data[key]

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __or__(self, other: DataMapping[HashableT, _T1] | dict[HashableT, _T1]) -> DataMapping[HashableT, _T1]:
        return DataMapping(self._data | (other._data if isinstance(other, DataMapping) else other))


class DataWorker(NamedAndSized, Mapping[HashableT, _T1], abc.ABC):
    def __init__(self, indices: Iterable[HashableT], **config: Any) -> None:
        super().__init__()
        self.indices: Final[list[HashableT]] = list(indices)
        self.attrs: Final[DataMapping[str, Any]] = DataMapping(config)

    @abc.abstractmethod
    def __getitem__(self, key: HashableT) -> _T1:
        ...

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self.indices)

    def __repr__(self) -> str:
        name = self.name
        size = self.size
        keys = repr_(self.indices, map_values=True)
        data = repr_(get_first_order_generic(self)[-1])
        text = join_kv(f"{name}({size=}):", *zip(keys, itertools.repeat(data)), start=0, stop=5)
        return text

    @property
    def name(self) -> str:
        name = super().name
        if split_name := self.attrs.get("split_name", None):
            name += f"[{split_name}]"
        return name

    def split(self, frac: float = 0.8, split_names=("train", "test")) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self.indices[:n], self.indices[n:]
        return (
            cls(indices=left, **self.attrs | {"split_name": split_names[0]}),
            cls(indices=right, **self.attrs | {"split_name": split_names[1]}),
        )

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self.indices)
        return self


# =====================================================================================================================
# - DataGenerator
# =====================================================================================================================
class DataGenerator(NamedAndSized, IterableDataset[_T1]):
    def __init__(
        self,
        worker: Mapping[HashableT, _T1] | Mapping[HashableT, _T1],
        *,
        maxsize: int = 0,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self.thread: Final[threading.Thread] = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue: Final[queue.Queue[_T1]] = queue.Queue(maxsize=maxsize)
        self.worker: Final[Mapping[HashableT, _T1]] = worker
        self.timeout: Final[float | None] = timeout

    def _target(self) -> None:
        for idx in self.worker.keys():
            self.queue.put(self.worker[idx], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[_T1]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.thread.start()
        return self
