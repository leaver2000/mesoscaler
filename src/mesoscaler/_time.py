from __future__ import annotations

import datetime

import numpy as np

from ._metadata import VariableEnum, auto_field
from ._typing import Any, Array, Literal, N, TypeAlias, overload

_auto_frequency = lambda x: auto_field(x, aliases=[f"datetime64[{x}]", f"timedelta64[{x}]"])
Falsy: TypeAlias = Literal[False, 0] | None
Truthy: TypeAlias = Literal[True, 1]


class Temporal(str, VariableEnum):
    year = _auto_frequency("Y")
    month = _auto_frequency("M")
    day = _auto_frequency("D")
    hour = _auto_frequency("h")
    minute = _auto_frequency("m")
    second = _auto_frequency("s")
    millisecond = _auto_frequency("ms")
    microsecond = _auto_frequency("us")
    nanosecond = _auto_frequency("ns")

    @classmethod
    def _missing_(cls, value) -> Temporal:
        if isinstance(value, np.dtype) and value.type is np.datetime64:
            return cls(value.name)
        raise ValueError(f"Invalid value for {cls.__class__.__name__}: {value!r}")

    @overload
    def arange(
        self,
        start: datetime.datetime | np.datetime64 | str,
        stop: datetime.datetime | np.datetime64 | str | None = None,
        step: datetime.timedelta | np.timedelta64 | int | None = None,
    ) -> Array[[N], np.datetime64]:
        ...

    @overload
    def arange(
        self,
        start: np.timedelta64 | datetime.timedelta | int,
        stop: np.timedelta64 | datetime.timedelta | int | None = None,
        step: datetime.timedelta | np.timedelta64 | int | None = None,
    ) -> Array[[N], np.timedelta64]:
        ...

    def arange(
        self,
        start: datetime.datetime | np.datetime64 | str | np.timedelta64 | datetime.timedelta | int,
        stop: datetime.datetime | np.datetime64 | str | np.timedelta64 | datetime.timedelta | int | None = None,
        step: datetime.timedelta | np.timedelta64 | int | None = None,
    ) -> Array[[N], np.datetime64 | np.timedelta64]:
        # dtype = "dt" if isinstance(start, (datetime.datetime, np.datetime64, str)) else "td"
        x = isinstance(start, (datetime.datetime, np.datetime64, str))
        dtype = self.get_dtype(x)

        return np.arange(start, stop, step, dtype=dtype)

    def delta(self, value: int | datetime.timedelta | np.timedelta64) -> np.timedelta64:
        return np.timedelta64(value, self)

    @overload
    def datetime(
        self,
        __x: int | datetime.datetime | np.datetime64 | str | None = None,
    ) -> np.datetime64:
        ...

    @overload
    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = ...,
        minute: int = ...,
        second: int = ...,
        microsecond: int = ...,
        nanosecond: int = ...,
        /,
    ) -> np.datetime64:
        ...

    def datetime(self, year: int | datetime.datetime | np.datetime64 | str | None = None, *args: Any) -> np.datetime64:
        if args and isinstance(year, int):
            year = datetime.datetime(year, *args)
        return np.datetime64(year, self)

    @overload
    def get_dtype(self, timedelta: Falsy = ..., /) -> np.dtype[np.datetime64]:
        ...

    @overload
    def get_dtype(self, timedelta: Truthy, /) -> np.dtype[np.timedelta64]:
        ...

    @overload
    def get_dtype(self, timedelta: Falsy | Truthy | bool, /) -> np.dtype[np.datetime64 | np.timedelta64]:
        ...

    def get_dtype(self, timedelta: Falsy | Truthy | bool = None, /) -> np.dtype[np.datetime64 | np.timedelta64]:
        return np.dtype(self.aliases[bool(timedelta)])


TimeFrequencyLike: TypeAlias = (
    Temporal
    | np.dtype[np.datetime64]
    | Literal[
        "datetime64[Y]",
        "datetime64[M]",
        "datetime64[D]",
        "datetime64[h]",
        "datetime64[m]",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "Y",
        "M",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ]
)
