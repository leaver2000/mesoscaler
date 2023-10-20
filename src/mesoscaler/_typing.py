# mypy: ignore-errors
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
    "DependentDataset",
]
import datetime
import os
import sys
import typing
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

import enum
import types

from numpy._typing._nested_sequence import _NestedSequence as NestedSequence
from pandas._typing import Dtype

GenericAliasType: type[types.GenericAlias] = getattr(typing, "_GenericAlias", types.GenericAlias)
Undefined = enum.Enum("Undefined", names="_", module=__name__)

# =====================================================================================================================
_P = ParamSpec("_P")
_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_T_contra = TypeVar("_T_contra", bound=Any, covariant=True)
NumpyGeneric_T = TypeVar("NumpyGeneric_T", bound=np.generic, covariant=True)
NumpyDType_T = TypeVar("NumpyDType_T", bound=np.dtype[Any])
NumpyNumber_T = TypeVar("NumpyNumber_T", bound=np.number[Any])

Number_T = TypeVar("Number_T", bound="Number", covariant=True)
PandasDType_T = TypeVar(
    "PandasDType_T",
    bound=Union[
        Any,
        str,
        bytes,
        datetime.date,
        datetime.time,
        bool,
        int,
        float,
        complex,
        Dtype,
        datetime.datetime,  # includes pd.Timestamp
        datetime.timedelta,  # includes pd.Timedelta
        pd.Period,
        "pd.Interval[int | float | pd.Timestamp | pd.Timedelta]",
        pd.CategoricalDtype,
    ],
)

if TYPE_CHECKING:

    class Nd(Concatenate[_P]):
        ...

    PandasArrayLike: TypeAlias = Union[
        pd.Index[_T_co],
        pd.Series[_T_co],
    ]
    from .core import DependentDataset

else:
    Nd = Concatenate
    PandasArrayLike: TypeAlias = Union[
        pd.Index,
        pd.Series,
        list[_T_co],
    ]
    DependentDataset: TypeAlias = Any


NumberT = TypeVar("NumberT", bound="Number")
HashableT = TypeVar("HashableT", bound=Hashable)

# - builtins
Pair: TypeAlias = tuple[_T, _T]
DictStr: TypeAlias = dict[str, _T]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = "str | os.PathLike[str]"
Method: TypeAlias = Callable[Concatenate[_T, _P], _T_co]
ClassMethod: TypeAlias = Callable[Concatenate[type[_T], _P], _T_co]
ItemsType: TypeAlias = Mapping[_T, _T_co] | Iterator[tuple[_T, _T_co]] | Sequence[tuple[_T, _T_co]]

# - numpy
Int: TypeAlias = int | np.integer[Any]
Float: TypeAlias = float | np.floating[Any]
Number: TypeAlias = int | float
Boolean: TypeAlias = bool | np.bool_

# - pandas
PythonScalar: TypeAlias = Union[str, float, bool]
DatetimeLikeScalar: TypeAlias = Union["pd.Period", "pd.Timestamp", "pd.Timedelta"]
PandasScalar: TypeAlias = Union["pd.Period", "pd.Timestamp", "pd.Timedelta", "pd.Interval"]
Scalar: TypeAlias = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime.date]
MaskType: TypeAlias = Union["pd.Series[bool]", "NDArray[np.bool_]", list[bool]]


# =====================================================================================================================
# - Protocols
# =====================================================================================================================
def get_first_order_generic(obj: Any, default=(Undefined,)) -> tuple[types.GenericAlias, ...] | tuple[Undefined]:
    for arg in getattr(obj, "__orig_bases__", []):
        types_ = getattr(arg, "__args__", None)
        if types_ is None:
            return default
        return types_

    return default


class _Slice(Protocol[_T_co]):
    @property
    def start(self) -> _T_co:
        ...

    @property
    def stop(self) -> _T_co:
        ...

    @property
    def step(self) -> _T_co:
        ...


Slice: TypeAlias = _Slice[_T_co] | slice
TimeSlice: TypeAlias = Slice[np.datetime64]


class Indices(Sized, Iterable[_T_co], Protocol[_T_co]):
    ...


class Closeable(Protocol):
    def close(self) -> None:
        ...


class Shaped(Sized, Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...


Array: TypeAlias = np.ndarray[Nd[_P], np.dtype[NumpyGeneric_T]]
""">>> x: Array[[int, int], np.int_] = np.array([[1, 2, 3]]) # Array[(int, int), int]"""
NDArray: TypeAlias = Array[[...], NumpyGeneric_T]
List: TypeAlias = list[_T | Any]
TensorLike: TypeAlias = Union[Array[_P, _T_co], NDArray[_T_co]]

# - NewType
N = NewType(":", Any)
N1 = NewType("1", Any)
N2 = NewType("2", Any)
N3 = NewType("3", Any)
N4 = NewType("4", Any)

Nv = NewType("Nv", N)
Nx = NewType("Nx", N)
Ny = NewType("Ny", N)
Nt = NewType("Nt", N)
Nz = NewType("Nz", N)

if TYPE_CHECKING:
    from numpy._typing._array_like import _ArrayLike as ArrayLike
else:
    ArrayLike = np.ndarray[Any, _T_co]

# ArrayLike: TypeAlias = Union[np.ndarray[Any, _T_co], ExtensionArray]

AnyArrayLike: TypeAlias = Union[
    np.ndarray[Any, _T_contra],
    PandasArrayLike[PandasDType_T],
    list[PandasDType_T],
]

ListLike: TypeAlias = Sequence[_T_co] | Iterator[_T_co]
AreaExtent: TypeAlias = Array[[N4], np.float_]
"""A 4-tuple of `(x_min, y_min, x_max, y_max)`"""
Longitude: TypeAlias = float
Latitude: TypeAlias = float
Point: TypeAlias = tuple[Longitude, Latitude]
PointOverTime: TypeAlias = tuple[Point, TimeSlice | Array[[N], np.datetime64]]
