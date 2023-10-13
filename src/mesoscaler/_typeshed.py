# mypy: ignore-errors
from __future__ import annotations

__all__ = [
    "Nd",
    "Array",
    "ListLike",
    "ArrayLike",
    "AnyArrayLike",
    "NestedSequence",
    "TensorLike",
]
import typing

if typing.TYPE_CHECKING:
    import datetime
    from typing import Any, Concatenate, NewType, ParamSpec, TypeAlias, TypeVar, Union

    import numpy as np
    import pandas as pd
    import torch
    from numpy._typing._nested_sequence import _NestedSequence as NestedSequence
    from pandas._typing import Dtype
    from pandas.core.arrays.base import ExtensionArray

    _P = ParamSpec("_P")
    _T = TypeVar("_T", bound=Any)
    _T_co = TypeVar("_T_co", bound=Any, covariant=True)
    _T_contra = TypeVar("_T_contra", bound=Any, covariant=True)
    _Numpy_T_co = TypeVar("_Numpy_T_co", covariant=True, bound=np.generic)
    NumpyDType_T = TypeVar("NumpyDType_T", bound=np.dtype[Any])
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

    class Nd(Concatenate[_P]):
        ...

    # - TypeAlias
    Array: TypeAlias = np.ndarray[Nd[_P], np.dtype[_Numpy_T_co]]
    """>>> x: Array[[int,int], np.int_] = np.array([[1, 2, 3]]) # Array[(int,int), int]"""
    NDArray: TypeAlias = Array[..., _Numpy_T_co]
    List: TypeAlias = list[_T | Any]
    TensorLike: TypeAlias = Union[Array[_P, _T_co], NDArray[_T_co], torch.Tensor]

    # - NewType
    N = NewType(":", Any)
    N1 = NewType("1", Any)
    N2 = NewType("2", Any)
    N3 = NewType("3", Any)
    N4 = NewType("4", Any)

    ArrayLike: TypeAlias = Union[ExtensionArray, np.ndarray[Any, _T_co]]
    PandasArrayLike: TypeAlias = Union[pd.Index[_T_co], pd.Series[_T_co]]
    AnyArrayLike: TypeAlias = Union[ArrayLike[NumpyDType_T], PandasArrayLike[PandasDType_T], list[PandasDType_T]]  # type: ignore
    _ListLike = Union[AnyArrayLike[_T_contra, _T_contra], list[_T_contra]]
    ListLike: TypeAlias = _ListLike[_T | Any]


else:
    import numpy.typing as npt

    NumpyDType_T = typing.TypeVar("NumpyDType_T")
    PandasDType_T = typing.TypeVar("PandasDType_T")

    NestedSequence = typing.Sequence  # NestedSequence[int]
    Nd = typing.Tuple  # Nd[int, int, ...]
    Array = typing.Callable  # Array[[int,int], np.int_]
    NDArray = npt.NDArray
    ArrayLike = typing.List  # ArrayLike[int]
    PandasArrayLike = typing.List
    AnyArrayLike = typing.Tuple  # AnyArrayLike[int,int]
    List = typing.List
    ListLike = typing.List  # ListLike[int]
    TensorLike = typing.Callable  # TensorLike[[int,int], torch.int_]
    N = N1 = N2 = N3 = N4 = typing.Any
