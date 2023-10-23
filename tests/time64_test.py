import pytest
import numpy as np

from src.mesoscaler.time64 import Time64


@pytest.mark.parametrize(
    "start,stop,freq,dtype",
    [("2020-01-01T00:00:00.000000000", "2021-01-02T00:00:00.000000000", "M", "datetime64[M]")],
)
def test_date_range(start: str, stop: str, freq: str, dtype: str) -> None:
    expect = np.arange(start, stop, dtype=dtype)
    for item in (freq, dtype):
        assert np.all(expect == Time64(item).arange(start, stop))


def test_datetime():
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

    # assert TimeFrequency.hour.datetime(2023, 1, 1).dtype == TimeFrequency.hour.dt.dtype


#     print(TimeFrequency.hour.dt.dtype)
#     print(TimeFrequency.hour.aliases)


# def test_timedelta() -> None:
#     for x in TimeFrequency:
#         td = x.timedelta(1)
#         assert isinstance(td, np.timedelta64)
