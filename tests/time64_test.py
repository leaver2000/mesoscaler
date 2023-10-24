from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.mesoscaler.time64 as time64
from src.mesoscaler.time64 import Time64


@pytest.mark.parametrize(
    "expect,name,value,dt_dtype,td_dtype",
    [
        (time64.years, "years", "Y", "datetime64[Y]", "timedelta64[Y]"),
        (time64.months, "months", "M", "datetime64[M]", "timedelta64[M]"),
        (time64.days, "days", "D", "datetime64[D]", "timedelta64[D]"),
        (time64.hours, "hours", "h", "datetime64[h]", "timedelta64[h]"),
        (time64.minutes, "minutes", "m", "datetime64[m]", "timedelta64[m]"),
        (time64.seconds, "seconds", "s", "datetime64[s]", "timedelta64[s]"),
        (time64.milliseconds, "milliseconds", "ms", "datetime64[ms]", "timedelta64[ms]"),
        (time64.microseconds, "microseconds", "us", "datetime64[us]", "timedelta64[us]"),
    ],
)
def test_time64_is(expect: Time64, name: str, value: str, dt_dtype, td_dtype) -> None:
    assert (
        expect
        is Time64[name]
        is Time64(name)
        is Time64(value)
        is Time64(dt_dtype)
        is Time64(td_dtype)
        is Time64.loc[name]  # type: ignore
    )


@pytest.mark.parametrize(
    "dt_dtype,td_dtype",
    [
        ("datetime64[Y]", "timedelta64[Y]"),
        ("datetime64[M]", "timedelta64[M]"),
        ("datetime64[D]", "timedelta64[D]"),
        ("datetime64[h]", "timedelta64[h]"),
        ("datetime64[m]", "timedelta64[m]"),
        ("datetime64[s]", "timedelta64[s]"),
        ("datetime64[ms]", "timedelta64[ms]"),
        ("datetime64[us]", "timedelta64[us]"),
        ("datetime64[ns]", "timedelta64[ns]"),
    ],
)
def test_time64_dtype(dt_dtype, td_dtype) -> None:
    assert (
        Time64(dt_dtype).dtypes
        == Time64(td_dtype).dtypes
        == (np.dtype(dt_dtype), np.dtype(td_dtype))
        == (dt_dtype, td_dtype)
    )


def test_time64_datetime() -> None:
    assert (
        Time64.hours.datetime(2023, 1, 1)
        == Time64("h").datetime(2023, 1, 1)
        == Time64("hours").datetime(2023, 1, 1)
        == Time64("datetime64[h]").datetime(2023, 1, 1)
        == Time64.hours.datetime("2023-01-01")
        == Time64.hours.datetime(Time64.hours.datetime("2023-01-01"))
        == np.datetime64("2023-01-01", "h")
    )
    assert Time64.hours.datetime(2023, 1, 1).astype(str) == "2023-01-01T00"


def f():
    assert (
        time64.hours.datetime("2022-01-01")
        == time64.hours.datetime(2022, 1, 1)
        == time64.hours.datetime(datetime.datetime(2022, 1, 1))
        == time64.hours.datetime(np.datetime64("2022-01-01"))
        == np.datetime64("2022-01-01", "h")
    )
    assert time64.hours.delta(2) == np.timedelta64(2, "h")


@pytest.mark.parametrize(
    "unit,obj,expect",
    [
        ("months", "2023-01-01", "datetime64[M]"),
        ("hours", datetime.datetime(2022, 1, 1), "datetime64[h]"),
        ("hours", np.datetime64("2022-01-01"), "datetime64[h]"),
        ("hours", pd.Timestamp("2022-01-01"), "datetime64[h]"),
        # - timedelta -
        ("hours", 1, "timedelta64[h]"),
    ],
)
def test_time64_infer_dtype(unit, obj, expect: str) -> None:
    assert time64.Time64(unit).infer_dtype(obj) == expect


@pytest.mark.parametrize(
    "start,stop,freq,dtype",
    [
        ("2020-01-01T00:00:00.000000000", "2021-01-02T00:00:00.000000000", "M", "datetime64[M]"),
        (1, 6, "hours", "timedelta64[h]"),
    ],
)
def test_time64_arange(start: Any, stop: Any, freq: str, dtype: str) -> None:
    expect = np.arange(start, stop, dtype=dtype)
    for item in (freq, dtype):
        assert np.all(expect == Time64(item).arange(start, stop))


@pytest.mark.parametrize(
    "t64,start,stop,dtype,size",
    [
        (Time64.hours, "2020-01-01", "2021-01-02", "datetime64[h]", 3),
        (Time64.hours, "2020-01-01", "2021-01-02", "datetime64[h]", 6),
        (Time64.days, "2020-01-01", "2022-01-02", "datetime64[D]", 3),
    ],
)
def test_time64_batch(t64: Time64, start: Any, stop: Any, dtype: str, size: int) -> None:
    expect = np.arange(start, stop, dtype=dtype).reshape(-1, size)
    assert np.all(expect == t64.batch(start, stop, size=size))
