from __future__ import annotations

import datetime

import numpy as np

from ._metadata import VariableEnum, auto_field
from ._typing import Any, Array, Literal, N, TypeAlias, Union, overload

TimeDeltaValue: TypeAlias = datetime.timedelta | np.timedelta64 | int
DateTimeValue: TypeAlias = datetime.datetime | np.datetime64 | str | float
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


_auto_frequency = lambda x: auto_field(x, aliases=[f"datetime64[{x}]", f"timedelta64[{x}]"])


class Time64(str, VariableEnum):
    years = _auto_frequency("Y")
    months = _auto_frequency("M")
    days = _auto_frequency("D")
    hours = _auto_frequency("h")
    minutes = _auto_frequency("m")
    seconds = _auto_frequency("s")
    milliseconds = _auto_frequency("ms")
    microseconds = _auto_frequency("us")
    nanoseconds = _auto_frequency("ns")

    @classmethod
    def _missing_(cls, value) -> Time64:
        if isinstance(value, np.dtype) and value.type in (np.datetime64, np.timedelta64):
            return cls(value.name)
        raise ValueError(f"Invalid value for {cls.__class__.__name__}: {value!r}")

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
        return np.dtype(self.aliases[~isinstance(x, (datetime.datetime, np.datetime64, float, str))])

    @overload
    def arange(
        self, start: DateTimeValue, stop: DateTimeValue, step: TimeDeltaValue | None = None
    ) -> Array[[N], np.datetime64]:
        ...

    @overload
    def arange(
        self, start: TimeDeltaValue, stop: TimeDeltaValue, step: TimeDeltaValue | None = None
    ) -> Array[[N], np.timedelta64]:
        ...

    def arange(
        self,
        start: DateTimeValue | TimeDeltaValue,
        stop: DateTimeValue | TimeDeltaValue,
        step: TimeDeltaValue | None = None,
    ) -> Array[[N], np.datetime64 | np.timedelta64]:
        dtype = self.infer_dtype(start)

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

    def datetime(self, __x: int | datetime.datetime | np.datetime64 | str | None = None, *args: Any) -> np.datetime64:
        if args and isinstance(__x, int):
            __x = datetime.datetime(__x, *args)
        return np.datetime64(__x, self)


years, months, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds = (
    Time64.years,
    Time64.months,
    Time64.days,
    Time64.hours,
    Time64.minutes,
    Time64.seconds,
    Time64.milliseconds,
    Time64.microseconds,
    Time64.nanoseconds,
)


def daterange(
    start: DateTimeValue,
    end: DateTimeValue,
    step: TimeDeltaValue | None = None,
    /,
    *,
    freq: Time64Literal | Time64 = hours,
) -> Array[[N], np.datetime64]:
    return Time64(freq).arange(start, end, step)