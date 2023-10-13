# noqa
from __future__ import annotations

__all__ = [
    "TYPE_CHECKING",
    "Any",
    "Callable",
    "ClassVar",
    "Generic",
    "Mapping",
    "cast",
    "Iterator",
    "Final",
    "TypedDict",
    "overload",
    "Annotated",
    "Collection",
    "EllipsisType",
    "Concatenate",
    "TypeGuard",
    "Scalar",
    "Hashable",
    "Literal",
    # - 3.10
    "TypeAlias",
    "ParamSpec",
    # - 3.11
    "Self",
    "Unpack",
    "TypeVarTuple",
    #
    "Array",
    "NDArray",
    "AnyArrayLike",
    "ArrayLike",
    "ListLike",
    "Nd",
    "N",
    "N1",
    "N2",
    "N3",
    "N4",
    "Number",
    "NumberT",
    "Boolean",
    "TensorLike",
    "NestedSequence",
    "TensorLike",
    "Sequence",
    "NewType",
    "MutableMapping",
    "NamedTuple",
    "NumpyDType_T",
    "PandasDType_T",
    "PandasArrayLike",
]
import datetime
import os
import sys


from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Collection,
    Concatenate,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    NewType,
    Protocol,
    Sequence,
    Sized,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd

from ._typeshed import (
    N1,
    N2,
    N3,
    N4,
    AnyArrayLike,
    Array,
    ArrayLike,
    ListLike,
    N,
    Nd,
    NDArray,
    NestedSequence,
    NumpyDType_T,
    PandasArrayLike,
    PandasDType_T,
    TensorLike,
)

if sys.version_info <= (3, 9):
    from typing_extensions import ParamSpec, Self, TypeAlias, TypeVarTuple, Unpack

    EllipsisType: TypeAlias = "ellipsis"  # noqa
elif sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import ParamSpec, TypeAlias

    from typing_extensions import Self, TypeVarTuple, Unpack
else:
    from types import EllipsisType
    from typing import ParamSpec, Self, TypeAlias, TypeVarTuple, Unpack

# =====================================================================================================================
_P = ParamSpec("_P")
_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)

HashableT = TypeVar("HashableT", bound=Hashable)

# - builtins
Method = Callable[Concatenate[_T, _P], _T_co]
ClassMethod = Callable[Concatenate[type[_T], _P], _T_co]
Pair: TypeAlias = tuple[_T, _T]
DictStr: TypeAlias = dict[str, _T]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = "str | os.PathLike[str]"

# - numpy
Number: TypeAlias = int | float | np.number[Any]
Boolean: TypeAlias = bool | np.bool_
NumberT = TypeVar("NumberT", int, float, np.number[Any])

# - pandas
PythonScalar: TypeAlias = Union[str, float, bool]
DatetimeLikeScalar: TypeAlias = Union["pd.Period", "pd.Timestamp", "pd.Timedelta"]
PandasScalar: TypeAlias = Union["pd.Period", "pd.Timestamp", "pd.Timedelta", "pd.Interval"]
Scalar: TypeAlias = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime.date]
MaskType: TypeAlias = Union["pd.Series[bool]", "NDArray[np.bool_]", list[bool]]


class SliceLike(Protocol[_T_co]):
    @property
    def start(self) -> _T_co:
        ...

    @property
    def stop(self) -> _T_co:
        ...

    @property
    def step(self) -> _T_co:
        ...


Slice: TypeAlias = SliceLike[_T_co] | slice
TimeSlice: TypeAlias = SliceLike[np.datetime64]


# =====================================================================================================================
# - Protocols
# =====================================================================================================================
class Indices(Sized, Iterable[_T_co], Protocol[_T_co]):
    ...


class Closeable(Protocol):
    def close(self) -> None:
        ...


class Shaped(Sized, Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...
