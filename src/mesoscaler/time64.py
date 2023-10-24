"""
This module combines the functionality of the `np.datetime64` and `np.timedelta64` dtypes into a single `Time64` enum
where each member represents a specific unit of time. This allows for a more intuitive and consistent API for
representing time and time deltas in NumPy arrays.
"""
from __future__ import annotations

import datetime

import numpy as np

from . import utils
from ._metadata import VariableEnum, auto_field
from ._typing import Any, Array, Literal, N, TypeAlias, Union, overload

DateTimeValue: TypeAlias = datetime.datetime | np.datetime64 | str
TimeDeltaValue: TypeAlias = datetime.timedelta | np.timedelta64 | int | float
Datetime64UnitLiteral: TypeAlias = Literal[
    "datetime64[Y]",
    "datetime64[M]",
    "datetime64[D]",
    "datetime64[h]",
    "datetime64[m]",
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
]
TimeDelta64UnitLiteral: TypeAlias = Literal[
    "timedelta64[Y]",
    "timedelta64[M]",
    "timedelta64[D]",
    "timedelta64[h]",
    "timedelta64[m]",
    "timedelta64[s]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[ns]",
]
Time64Literal: TypeAlias = Literal[
    "Y",
    "M",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "years",
    "months",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
]
Time64Like: TypeAlias = Union[
    "Time64",
    np.dtype[np.datetime64],
    np.dtype[np.timedelta64],
    Time64Literal,
    Datetime64UnitLiteral,
    TimeDelta64UnitLiteral,
]

_auto_time = lambda x: auto_field(x, aliases=[f"datetime64[{x}]", f"timedelta64[{x}]"])


class Time64(str, VariableEnum):
    years = _auto_time("Y")
    months = _auto_time("M")
    days = _auto_time("D")
    hours = _auto_time("h")
    minutes = _auto_time("m")
    seconds = _auto_time("s")
    milliseconds = _auto_time("ms")
    microseconds = _auto_time("us")
    nanoseconds = _auto_time("ns")

    @classmethod
    def unpack(cls, x: np.datetime64 | np.timedelta64) -> tuple[Time64, int]:
        """
        >>> import mesoscaler as ms
        >>> ms.hours.unpack(np.datetime64('2022-01-01T00','h'))
        (<Time64.hours: 'h'>, 2022)
        >>> ms.hours.unpack(np.timedelta64(2,'h'))
        (<Time64.hours: 'h'>, 2)
        """
        return cls(x.dtype.name), int(x.astype(int))

    @classmethod
    def _missing_(cls, value) -> Time64:
        if isinstance(value, np.dtype) and value.type in (np.datetime64, np.timedelta64):
            return cls(value.name)
        raise ValueError(f"Invalid value for {cls.__class__.__name__}: {value!r}")

    @property
    def dtypes(self) -> tuple[str, str]:
        return tuple(self.aliases[:2])  # type: ignore

    @overload
    def infer_dtype(self, x: DateTimeValue, /) -> np.dtype[np.datetime64]:
        ...

    @overload
    def infer_dtype(self, x: TimeDeltaValue, /) -> np.dtype[np.timedelta64]:
        ...

    def infer_dtype(self, x: DateTimeValue | TimeDeltaValue, /) -> np.dtype[np.datetime64 | np.timedelta64]:
        """
        Returns the NumPy dtype for the given input value.

        Parameters:
        x (DateTimeValue | TimeDeltaValue): The input value for which to determine the dtype.

        Returns:
        np.dtype[np.datetime64 | np.timedelta64]: The NumPy dtype for the input value.
        """
        idx = ~isinstance(x, (datetime.datetime, np.datetime64, str))
        return np.dtype(self.aliases[idx])

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

    def datetime(self, __x: int | datetime.datetime | np.datetime64 | str | None = None, *args: Any) -> np.datetime64:
        if args and isinstance(__x, int):
            __x = datetime.datetime(__x, *args)
        return np.datetime64(__x, self)

    # - array methods -
    @overload
    def arange(
        self, start: DateTimeValue, stop: DateTimeValue, step: TimeDeltaValue | None = None, /
    ) -> Array[[N], np.datetime64]:
        ...

    @overload
    def arange(
        self, start: TimeDeltaValue, stop: TimeDeltaValue, step: TimeDeltaValue | None = None, /
    ) -> Array[[N], np.timedelta64]:
        ...

    @overload
    def arange(
        self,
        start: DateTimeValue | TimeDeltaValue,
        stop: DateTimeValue | TimeDeltaValue,
        step: TimeDeltaValue | None = None,
        /,
    ) -> Array[[N], np.datetime64 | np.timedelta64]:
        ...

    def arange(
        self,
        start: DateTimeValue | TimeDeltaValue,
        stop: DateTimeValue | TimeDeltaValue,
        step: TimeDeltaValue | None = None,
        /,
    ) -> Array[[N], np.datetime64 | np.timedelta64]:
        dtype = self.infer_dtype(start)

        return np.arange(start, stop, step, dtype=dtype)

    @overload
    def batch(
        self,
        start: DateTimeValue,
        stop: DateTimeValue,
        step: TimeDeltaValue | None = None,
        /,
        *,
        size: int,
    ) -> Array[[N, N], np.datetime64]:
        ...

    @overload
    def batch(
        self,
        start: TimeDeltaValue,
        stop: TimeDeltaValue,
        step: TimeDeltaValue | None = None,
        /,
        *,
        size: int,
    ) -> Array[[N, N], np.timedelta64]:
        ...

    @overload
    def batch(
        self,
        start: DateTimeValue | TimeDeltaValue,
        stop: DateTimeValue | TimeDeltaValue,
        step: TimeDeltaValue | None = None,
        /,
        *,
        size: int,
    ) -> Array[[N, N], np.datetime64 | np.timedelta64]:
        ...

    def batch(
        self,
        start: DateTimeValue | TimeDeltaValue,
        stop: DateTimeValue | TimeDeltaValue,
        step: TimeDeltaValue | None = None,
        /,
        *,
        size: int,
    ) -> Array[[N, N], np.datetime64 | np.timedelta64]:
        """
        >>> import mesoscaler as ms
        >>> ms.hours.batch('2022-01-01', '2022-02-01', 6, size=4)
        array([['2022-01-01T00', '2022-01-01T06', '2022-01-01T12','2022-01-01T18']
                ...
                ['2022-01-31T00', '2022-01-31T06', '2022-01-31T12', '2022-01-31T18']], dtype='datetime64[h]')
        """
        return utils.batch(self.arange(start, stop, step), size, strict=True)


del _auto_time


Y: Literal[Time64.years]
years: Literal[Time64.years]
M: Literal[Time64.months]
months: Literal[Time64.months]
D: Literal[Time64.days]
days: Literal[Time64.days]
h: Literal[Time64.hours]
hours: Literal[Time64.hours]
m: Literal[Time64.minutes]
minutes: Literal[Time64.minutes]
s: Literal[Time64.seconds]
seconds: Literal[Time64.seconds]
ms: Literal[Time64.milliseconds]
milliseconds: Literal[Time64.milliseconds]
us: Literal[Time64.microseconds]
microseconds: Literal[Time64.microseconds]
ns: Literal[Time64.nanoseconds]
nanoseconds: Literal[Time64.nanoseconds]


def __getattr__(name: str) -> Time64:
    """
    >>> import mesoscaler.time64 as t64
    >>> t64.hours
    <Time64.hours: 'h'>
    >>> t64.h
    <Time64.hours: 'h'>
    >>> t64.h.datetime('2001-01-01')
    numpy.datetime64('2001-01-01T00','h')
    >>> t64.h.delta(2)
    numpy.timedelta64(2,'h')
    >>> t64.h.batch('2023-01-01', '2023-01-07', 6, size=4)
    array([['2023-01-01T00', '2023-01-01T06', '2023-01-01T12', '2023-01-01T18'],
           ['2023-01-02T00', '2023-01-02T06', '2023-01-02T12', '2023-01-02T18'],
           ['2023-01-03T00', '2023-01-03T06', '2023-01-03T12', '2023-01-03T18'],
           ['2023-01-04T00', '2023-01-04T06', '2023-01-04T12', '2023-01-04T18'],
           ['2023-01-05T00', '2023-01-05T06', '2023-01-05T12', '2023-01-05T18'],
           ['2023-01-06T00', '2023-01-06T06', '2023-01-06T12', '2023-01-06T18']],
           dtype='datetime64[h]')
    """
    if name in __annotations__:
        return Time64(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
