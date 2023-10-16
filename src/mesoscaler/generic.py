"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import bisect
import itertools
import queue
import random
import threading
from typing import ForwardRef

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
    Sized,
    TypeVar,
    TypeVarTuple,
    get_first_order_generic,
    overload,
)
from .utils import acc_size, is_array_like, join_kv

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts")


class NamedAndSized(Sized, abc.ABC):
    __slots__ = ()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def size(self) -> int:
        return len(self)


class GenericRepresentation:
    def __init__(self, x: Any) -> None:
        x = get_first_order_generic(x)[-1]
        if isinstance(x, ForwardRef):
            x = x.__forward_arg__

        self.x = (
            str(x)
            .replace(getattr(x, "__module__", "") + ".", "")
            .replace(f"{__package__}.", "")
            .replace("__main__.", "")
            .replace("_typing.", "")
            .replace("typing.", "")
            .replace("Ellipsis", "...")
        )

    def __repr__(self) -> str:
        return self.x


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
        name = self.name
        data = self.data
        size = self.size
        text = join_kv(f"{name}({size=}):", *data, start=0, stop=5)
        return text

    def to_dict(self) -> dict[Hashable, _T]:
        return dict(self.data)


class MappingBase(NamedAndSized, Mapping[HashableT, _T], abc.ABC):
    @property
    def data(self) -> Iterable[tuple[HashableT, _T]]:
        return self.items()


# =====================================================================================================================
#
# =====================================================================================================================
class Dataset(NamedAndSized, Generic[_T], abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> _T:
        ...

    def __add__(self, other: Dataset[_T]) -> ConcatDataset[_T]:
        return ConcatDataset([self, other])


class ConcatDataset(Dataset[_T]):
    def __init__(self, data: Iterable[Dataset[_T]]) -> None:
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

    def __getitem__(self, idx: int) -> _T:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)

        if ds_idx := bisect.bisect_right(self.accumulated_sizes, idx):
            idx -= self.accumulated_sizes[ds_idx - 1]

        return self.data[ds_idx][idx]


# =====================================================================================================================
#
# =====================================================================================================================
class IterableDataset(NamedAndSized, Generic[_T], abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[_T]:
        ...

    def __add__(self, other: IterableDataset[_T]) -> ChainDataset[_T]:
        return ChainDataset([self, other])


class ChainDataset(IterableDataset[_T]):
    def __init__(self, datasets: Iterable[IterableDataset[_T]]) -> None:
        super().__init__()
        self.data = datasets

    def __iter__(self) -> Iterator[_T]:
        return itertools.chain.from_iterable(self.data)

    def __len__(self) -> int:
        return sum(map(len, self.data))


# =====================================================================================================================
# - Mappings
# =====================================================================================================================
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


class DataWorker(MappingBase[HashableT, _T]):
    def __init__(self, indices: Iterable[HashableT], **config: Any) -> None:
        super().__init__()
        self.indices: Final[list[HashableT]] = list(indices)
        self.config: Final[DictStrAny] = config

    @abc.abstractmethod
    def __getitem__(self, key: HashableT) -> _T:
        ...

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self.indices)

    def __repr__(self) -> str:
        name = self.name
        size = self.size
        # because we're lazily loading data we can attempt to use the type annotation of _T class as the representation
        data = GenericRepresentation(self)
        text = join_kv(f"{name}({size=}):", *zip(self.indices, itertools.repeat(data)), start=0, stop=5)
        return text

    @property
    def name(self) -> str:
        name = super().name
        if split_name := self.config.get("split_name", None):
            name += f"[{split_name}]"
        return name

    def split(self, frac: float = 0.8, split_names=("train", "test")) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self.indices[:n], self.indices[n:]
        return (
            cls(indices=left, **self.config | {"split_name": split_names[0]}),
            cls(indices=right, **self.config | {"split_name": split_names[1]}),
        )

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self.indices)
        return self


# =====================================================================================================================
#
# =====================================================================================================================
class DataConsumer(IterableDataset[_T], Generic[HashableT, _T]):
    def __init__(self, worker: Mapping[HashableT, _T], *, maxsize: int = 0, timeout: float | None = None) -> None:
        super().__init__()
        self.thread: Final[threading.Thread] = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue: Final[queue.Queue[_T]] = queue.Queue[_T](maxsize=maxsize)
        self.worker: Final[Mapping[HashableT, _T]] = worker
        self.timeout: Final[float | None] = timeout

    def _target(self) -> None:
        for idx in self.worker.keys():
            self.queue.put(self.worker[idx], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[_T]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.thread.start()
        return self


# =====================================================================================================================
#
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
