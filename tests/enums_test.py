import numpy as np
import pandas as pd
import pytest

from src.mesoscaler._metadata import (
    _CLASS_METADATA,
    _MEMBER_METADATA,
    VariableEnum,
    auto_field,
)
from src.mesoscaler.enums import (
    ERA5,
    LAT,
    LON,
    LVL,
    TIME,
    URMA,
    Dimensions,
    T,
    TimeFrequency,
    X,
    Y,
    Z,
)


def test_coordinate_axes() -> None:
    assert LVL.axis == (Z,)
    assert TIME.axis == (T,)
    assert LAT.axis == (Y, X)
    assert LON.axis == (Y, X)


def test_main():
    with pytest.raises(ValueError):
        ERA5("nope")

    assert ERA5.Z == "geopotential"
    assert ERA5("z") == "geopotential"
    assert ERA5(["z"]) == ["geopotential"]
    assert ERA5(["z", "q"]) == ["geopotential", "specific_humidity"]
    assert ERA5.to_list() == [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]

    assert URMA.loc[["TCC"]] == ["total_cloud_cover"]
    assert isinstance(Dimensions.to_frame(), pd.DataFrame)
    assert LAT.axis == (Y, X)
    assert ERA5("z") is ERA5.Z and ERA5.Z is ERA5("geopotential") and ERA5.Z == "geopotential"
    assert ERA5("z") == ERA5.Z
    assert set(ERA5.difference(list("tuv"))) == set(ERA5).difference(ERA5(list("tuv")))


class MyEnum(str, VariableEnum, my_class_metadata="hello"):
    A = auto_field("a", aliases=["alpha"], hello="world")
    B = auto_field("b", aliases=["beta"])
    C = auto_field("c", aliases=["beta"])
    D = auto_field("d", aliases=[4, 5, 6])


def test_my_enum() -> None:
    assert MyEnum.A == "a"
    assert MyEnum.loc[["A", "B"]] == [MyEnum.A, MyEnum.B]
    assert MyEnum["A"] == "a" == MyEnum.A == MyEnum("alpha")


def test_my_enum_metadata() -> None:
    assert MyEnum.__metadata__["name"] == "MyEnum"


def test_my_enum_class_metadata() -> None:
    class_meta = MyEnum.__metadata__[_CLASS_METADATA]
    assert class_meta is MyEnum.metadata


def test_member_metadata() -> None:
    member_meta = MyEnum.A.metadata
    assert member_meta is MyEnum.__metadata__[_MEMBER_METADATA]["A"]

    mm = MyEnum._member_metadata
    assert MyEnum.__metadata__[_MEMBER_METADATA] is mm


def test_datetime():
    assert (
        TimeFrequency.hour.datetime(2023, 1, 1)
        == TimeFrequency("h").datetime(2023, 1, 1)
        == TimeFrequency("hour").datetime(2023, 1, 1)
        == TimeFrequency("datetime64[h]").datetime(2023, 1, 1)
        == TimeFrequency.hour.datetime("2023-01-01")
        == TimeFrequency.hour.datetime(TimeFrequency.hour.datetime("2023-01-01"))
        == np.datetime64("2023-01-01", "h")
    )
    assert TimeFrequency.hour.datetime(2023, 1, 1).astype(str) == "2023-01-01T00"

    # assert TimeFrequency.hour.datetime(2023, 1, 1).dtype == TimeFrequency.hour.dt.dtype
    print(TimeFrequency.hour.dt.dtype)
    print(TimeFrequency.hour.aliases)


def test_timedelta() -> None:
    for x in TimeFrequency:
        td = x.timedelta(1)
        assert isinstance(td, np.timedelta64)
