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
    "DependentDataset",
]
import datetime
import enum
import os
import sys
import types
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

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
from numpy._typing._nested_sequence import _NestedSequence as NestedSequence
from numpy.typing import NDArray

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

_P = ParamSpec("_P")
_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)


if TYPE_CHECKING:

    class Nd(Concatenate["_P"]):
        ...

    from numpy._typing._array_like import _ArrayLike as ArrayLike

    from .core import DependentDataset

else:
    Nd = Concatenate
    DependentDataset: TypeAlias = Any
    ArrayLike = np.ndarray[Any, _T_co]


GenericAliasType: type[types.GenericAlias] = getattr(typing, "_GenericAlias", types.GenericAlias)
Undefined = enum.Enum("Undefined", names="_", module=__name__)

# =====================================================================================================================

NumpyGeneric_T = TypeVar("NumpyGeneric_T", bound=np.generic | Any)
NumpyDType_T = TypeVar("NumpyDType_T", bound=np.dtype[Any])
NumpyNumber_T = TypeVar("NumpyNumber_T", bound=np.number[Any])

Number_T = TypeVar("Number_T", bound="Number")


NumberT = TypeVar("NumberT", bound="Number")
HashableT = TypeVar("HashableT", bound=Hashable)

# - builtins
Pair: TypeAlias = tuple[_T, _T]
Trips: TypeAlias = tuple[_T, _T, _T]
Quad: TypeAlias = tuple[_T, _T, _T, _T]

DictStr: TypeAlias = dict[str, _T]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = "str | os.PathLike[str]"
Method: TypeAlias = Callable[Concatenate[_T, _P], _T_co]
ClassMethod: TypeAlias = Callable[Concatenate[type[_T], _P], _T_co]
ChainableItems: TypeAlias = Mapping[_T, _T_co] | list[tuple[_T, _T_co]] | Iterator[tuple[_T, _T_co]]

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
List: TypeAlias = list[_T | Any]
TensorLike: TypeAlias = Union[Array[_P, _T_co], NDArray[_T_co]]

# - NewType
N = NewType("N", int)
Nv = NewType("Nv", int)
Nt = NewType("Nt", int)
Nz = NewType("Nz", int)
Ny = NewType("Ny", int)
Nx = NewType("Nx", int)
N1 = NewType("1", int)
N2 = NewType("2", int)
N3 = NewType("3", int)
N4 = NewType("4", int)


AnyArrayLike: TypeAlias = Array[[...], NumpyGeneric_T] | xr.DataArray | dask.array.Array

ListLike: TypeAlias = Sequence[_T_co] | Iterator[_T_co]
AreaExtent: TypeAlias = Array[[N4], np.float_]
"""A 4-tuple of `(x_min, y_min, x_max, y_max)`"""
Longitude: TypeAlias = float
Latitude: TypeAlias = float
Point: TypeAlias = tuple[Longitude, Latitude]
PointOverTime: TypeAlias = tuple[Point, Array[[N], np.datetime64]]
