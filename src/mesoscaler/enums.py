"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import datetime
import enum
import functools

import numpy as np
import pyproj

from ._metadata import (
    DependentVariables,
    IndependentVariables,
    VariableEnum,
    auto_field,
)
from ._typing import Any, Array, Literal, Mapping, N, TypeAlias, TypeVar, overload

_T = TypeVar("_T")
__NoDefault = enum.Enum("", "NoDefault")
NoDefault = __NoDefault.NoDefault
LiteralNoDefault = Literal[__NoDefault.NoDefault]
del __NoDefault


# =====================================================================================================================
#
# =====================================================================================================================
class Dimensions(IndependentVariables):
    T = auto_field(aliases=["t", "time"])
    Z = auto_field(aliases=["z", "level", "height", "altitude"])
    Y = auto_field(aliases=["y", "latitude", "grid_latitude"])
    X = auto_field(aliases=["x", "longitude", "grid_longitude"])


def unpack_dims() -> tuple[Literal[Dimensions.T], Literal[Dimensions.Z], Literal[Dimensions.Y], Literal[Dimensions.X]]:
    """>>> T, Z, Y, X = mesoscaler.unpack_dims()"""
    return (
        Dimensions.T,
        Dimensions.Z,
        Dimensions.Y,
        Dimensions.X,
    )


DIMENSIONS = T, Z, Y, X = unpack_dims()
DimensionsMapType: TypeAlias = Mapping[tuple[Literal[X], Literal[Y]] | Literal[Z] | Literal[T], _T]  # type: ignore[valid-type]


class Coordinates(IndependentVariables):
    time = auto_field(axis=(T,))
    vertical = auto_field(aliases=["level", "height"], axis=(Z,))
    latitude = auto_field(aliases=["grid_latitude"], axis=(Y, X))
    longitude = auto_field(aliases=["grid_longitude"], axis=(Y, X))

    @property
    def axis(self) -> tuple[Dimensions, ...]:
        return self.metadata["axis"]


def unpack_coords() -> (
    tuple[
        Literal[Coordinates.time],
        Literal[Coordinates.vertical],
        Literal[Coordinates.latitude],
        Literal[Coordinates.longitude],
    ]
):
    """>>> TIME, LEVEL, LAT, LON = mesoscaler.unpack_coords()"""
    return (
        Coordinates.time,
        Coordinates.vertical,
        Coordinates.latitude,
        Coordinates.longitude,
    )


COORDINATES = TIME, LVL, LAT, LON = unpack_coords()


# =====================================================================================================================
#
# =====================================================================================================================
class CoordinateReferenceSystem(functools.partial, enum.Enum):
    lambert_azimuthal_equal_area = laea = pyproj.CRS, {
        "proj": "laea",
    }
    lambert_conformal_conic = lcc = pyproj.CRS, {
        "proj": "lcc",
    }

    def from_point(self, longitude: float, latitude: float) -> pyproj.CRS:
        return self(longitude=longitude, latitude=latitude)


LiteralProjection = Literal["laea", "lcc", "lambert_azimuthal_equal_area", "lambert_conformal_conic"]
LiteralCRS = CoordinateReferenceSystem | LiteralProjection


class TimeFrequency(str, VariableEnum):
    Y = auto_field("datetime64[Y]", aliases=["year"])
    M = auto_field("datetime64[M]", aliases=["month"])
    D = auto_field("datetime64[D]", aliases=["day"])
    h = auto_field("datetime64[h]", aliases=["hour"])
    m = auto_field("datetime64[m]", aliases=["minute"])
    s = auto_field("datetime64[s]", aliases=["second"])
    ms = auto_field("datetime64[ms]", aliases=["millisecond"])
    us = auto_field("datetime64[us]", aliases=["microsecond"])
    ns = auto_field("datetime64[ns]", aliases=["nanosecond"])

    @property
    def dtype(self) -> np.dtype[np.datetime64]:
        return np.dtype(self)

    def arange(
        self,
        start: datetime.datetime | np.datetime64 | str,
        stop: datetime.datetime | np.datetime64 | str | None = None,
        step: int | datetime.timedelta | np.timedelta64 | None = None,
    ) -> Array[[N], np.datetime64]:
        return np.arange(start, stop, step, dtype=self.dtype)

    @classmethod
    def _missing_(cls, value) -> TimeFrequency:
        if isinstance(value, np.dtype) and value.type is np.datetime64:
            return cls(value.name)
        raise ValueError(f"Invalid value for {cls.__class__.__name__}: {value!r}")

    def timedelta(self, value: int | datetime.timedelta | np.timedelta64) -> np.timedelta64:
        return np.timedelta64(value, self.name)

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
        month: int = ...,
        day: int = ...,
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
        return np.datetime64(year, self.name)  # type: ignore


TimeFrequencyLike = (
    TimeFrequency
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

year, month, day, hour, minute, second, millisecond, microsecond, nanosecond = (
    TimeFrequency.Y,
    TimeFrequency.M,
    TimeFrequency.D,
    TimeFrequency.h,
    TimeFrequency.m,
    TimeFrequency.s,
    TimeFrequency.ms,
    TimeFrequency.us,
    TimeFrequency.ns,
)


# =====================================================================================================================
#
# =====================================================================================================================
class ERA5(DependentVariables):
    r"""
    | member_name   | short_name   | standard_name       | long_name           | type_of_level   | units      |
    |:--------------|:-------------|:--------------------|:--------------------|:----------------|:-----------|
    | Z             | z            | geopotential        | Geopotential        | isobaricInhPa   | m**2 s**-2 |
    | Q             | q            | specific_humidity   | Specific humidity   | isobaricInhPa   | kg kg**-1  |
    | T             | t            | temperature         | Temperature         | isobaricInhPa   | K          |
    | U             | u            | u_component_of_wind | U component of wind | isobaricInhPa   | m s**-1    |
    | V             | v            | v_component_of_wind | V component of wind | isobaricInhPa   | m s**-1    |
    | W             | w            | vertical_velocity   | Vertical velocity   | isobaricInhPa   | Pa s**-1   |
    """

    Z = auto_field("geopotential", aliases=["z"], units="m**2 s**-2")
    Q = auto_field("specific_humidity", aliases=["q"], units="kg kg**-1")
    T = auto_field("temperature", aliases=["t"], units="K")
    U = auto_field("u_component_of_wind", aliases=["u"], units="m s**-1")
    V = auto_field("v_component_of_wind", aliases=["v"], units="m s**-1")
    W = auto_field("vertical_velocity", aliases=["w"], units="Pa s**-1")

    @property
    def units(self) -> str:
        return self.metadata["units"]


ERA5_VARS = (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    VERTICAL_VELOCITY,
) = (
    ERA5.Z,
    ERA5.Q,
    ERA5.T,
    ERA5.U,
    ERA5.V,
    ERA5.W,
)


class URMA(DependentVariables):
    """
    | member_name   | short_name   | standard_name           | long_name                    | type_of_level         | units       |
    |:--------------|:-------------|:------------------------|:-----------------------------|:----------------------|:------------|
    | CEIL          | ceil         | ceiling                 | cloud ceiling                | cloudCeiling          | m           |
    | D2M           | d2m          | dewpoint_temperature_2m | 2 meter dewpoint temperature | heightAboveGround     | K           |
    | SH2           | sh2          | specific_humidity_2m    | 2 meter specific humidity    | heightAboveGround     | kg kg**-1   |
    | SP            | sp           | surface_pressure        | surface pressure             | surface               | Pa          |
    | T2M           | t2m          | temperature_2m          | 2 meter temperature          | heightAboveGround     | K           |
    | TCC           | tcc          | total_cloud_cover       | total cloud cover            | atmosphereSingleLayer | %           |
    | U10           | u10          | u_wind_component_10m    | 10 meter u wind component    | heightAboveGround     | m s**-1     |
    | V10           | v10          | v_wind_component_10m    | 10 meter v wind component    | heightAboveGround     | m s**-1     |
    | VIS           | vis          | visibility              | visibility                   | surface               | m           |
    | WDIR10        | wdir10       | wind_direction_10m      | 10 meter wind direction      | heightAboveGround     | Degree true |
    | SI10          | si10         | wind_speed_10m          | 10 meter wind speed          | heightAboveGround     | m s**-1     |
    | GUST          | gust         | wind_speed_gust         | wind speed gust              | heightAboveGround     | m s**-1     |
    | OROG          | orog         | orography               | surface orography            | surface               | m           |
    """

    TCC = auto_field("total_cloud_cover", units="%", type_of_level="atmosphereSingleLayer")
    CEIL = auto_field("ceiling", units="m", type_of_level="cloudCeiling")
    U10 = auto_field("u_wind_component_10m", units="m s**-1", type_of_level="heightAboveGround")
    V10 = auto_field("v_wind_component_10m", units="m s**-1", type_of_level="heightAboveGround")
    SI10 = auto_field("wind_speed_10m", units="m s**-1", type_of_level="heightAboveGround")
    GUST = auto_field("wind_speed_gust", units="m s**-1", type_of_level="heightAboveGround")
    WDIR10 = auto_field("wind_direction_10m", units="Degree true", type_of_level="heightAboveGround")
    T2M = auto_field("temperature_2m", units="K", type_of_level="heightAboveGround")
    D2M = auto_field("dewpoint_temperature_2m", units="K", type_of_level="heightAboveGround")
    SH2 = auto_field("specific_humidity_2m", units="kg kg**-1", type_of_level="heightAboveGround")
    SP = auto_field("surface_pressure", units="Pa", type_of_level="surface")
    VIS = auto_field("visibility", units="m", type_of_level="surface")
    OROG = auto_field("orography", units="m", type_of_level="surface")

    @property
    def units(self) -> str:
        return self.metadata["units"]

    @property
    def type_of_level(self) -> str:
        return self.metadata["type_of_level"]


URMA_VARS = (
    TOTAL_CLOUD_COVER,
    CEILING,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    WIND_SPEED_10M,
    WIND_SPEED_GUST,
    WIND_DIRECTION_10M,
    TEMPERATURE_2M,
    DEWPOINT_TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    VISIBILITY,
    OROGRAPHY,
) = (
    URMA.TCC,
    URMA.CEIL,
    URMA.U10,
    URMA.V10,
    URMA.SI10,
    URMA.GUST,
    URMA.WDIR10,
    URMA.T2M,
    URMA.D2M,
    URMA.SH2,
    URMA.SP,
    URMA.VIS,
    URMA.OROG,
)
