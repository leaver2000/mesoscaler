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
