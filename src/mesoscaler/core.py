from __future__ import annotations

from typing import NewType

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from pyresample.geometry import GridDefinition
from xarray.core.coordinates import DatasetCoordinates

from ._typing import (
    N2,
    N4,
    Any,
    Array,
    Hashable,
    Iterable,
    ListLike,
    Literal,
    Mapping,
    N,
    NDArray,
    Number,
    Self,
    Sequence,
    TimeSlicePoint,
    TypeAlias,
    Union,
)
from .enums import (
    DIMENSIONS,
    LAT,
    LON,
    LVL,
    TIME,
    Coordinates,
    DependentVariables,
    Dimensions,
    LiteralCRS,
    T,
    X,
    Y,
    Z,
)
from .generic import Data, DataWorker
from .sampling.resampler import AbstractInstructor, ReSampleInstructor, ReSampler
from .utils import log_scale, sort_unique

Nv = NewType("Nv", int)
Nx = NewType("Nx", int)
Ny = NewType("Ny", int)
Nt = NewType("Nt", int)
Nz = NewType("Nz", int)

Depends: TypeAlias = Union[type[DependentVariables], DependentVariables, Sequence[DependentVariables], "Dependencies"]
# ResampleInstruction: TypeAlias = tuple["DependentDataset", AreaExtent]
Unit = Literal["km", "m"]
# =====================================================================================================================
STANDARD_SURFACE_PRESSURE = P0 = 1013.25  # - hPa
P1 = 25.0  # - hPa
DEFAULT_PRESSURE: ListLike[Number] = [P0, 925.0, 850.0, 700.0, 500.0, 300.0]  # - hPa
DERIVED_SURFACE_COORDINATE = {LVL: (LVL.axis, [STANDARD_SURFACE_PRESSURE])}
"""If the Dataset does not contain a vertical coordinate, it is assumed to be a derived atmospheric parameter
or near surface parameter. The vertical coordinate is then set to the standard surface pressure of 1013.25 hPa."""
MESOSCALE_BETA = 200.0  # - km

_units: Mapping[Unit, float] = {"km": 1.0, "m": 1000.0}
_VARIABLES = "variables"
_GRID_DEFINITION = "grid_definition"
_DEPENDS = "depends"


# =====================================================================================================================
# - tools for preparing to data into a common format and convention
# =====================================================================================================================
class Dependencies:
    @staticmethod
    def _validate_variables(depends: Depends) -> tuple[type[DependentVariables], list[DependentVariables]]:
        if isinstance(depends, Dependencies):
            return depends.enum, depends.depends
        elif isinstance(depends, type):
            assert issubclass(depends, DependentVariables)
            enum = depends
            depends = list(depends)
        elif isinstance(depends, DependentVariables):
            enum = depends.__class__
            depends = [depends]
        else:
            enum = depends[0].__class__
            depends = list(depends)

        for dvar in depends:
            assert isinstance(dvar, enum)
        return enum, depends

    def __init__(self, depends: Depends):
        self.enum, self.depends = self._validate_variables(depends)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.enum.__name__})"

    @property
    def difference(self) -> set[DependentVariables]:
        return self.enum.difference(self.depends)

    @property
    def names(self) -> pd.Index[str]:
        return self.enum._names  # TODO: the enum names should not be a private attribute

    @property
    def crs(self) -> pyproj.CRS:
        return self.enum.crs  # type: ignore

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.enum.metadata  # type: ignore

    @property
    def name(self) -> str:
        return self.enum.name


def is_dimension_independent(dims: Iterable[Hashable]) -> bool:
    return all(isinstance(dim, Dimensions) for dim in dims) and set(dims) == set(Dimensions)


def is_coordinate_independent(coords: DatasetCoordinates) -> bool:
    return (
        all(isinstance(coord, Coordinates) for coord in coords)
        and set(coords) == set(Coordinates)
        and all(coords[x].dims == (Y, X) for x in (LON, LAT))
    )


def is_independent(ds: xr.Dataset) -> bool:
    return is_coordinate_independent(ds.coords) and is_dimension_independent(ds.dims)


def make_independent(ds: xr.Dataset) -> xr.Dataset:
    """insures a dependant dataset is in the correct format."""
    if is_independent(ds):
        return ds
    #  - rename the dims and coordinates
    ds = ds.rename_dims(Dimensions.remap(ds.dims)).rename_vars(Coordinates.remap(ds.coords))
    # - move any coordinates assigned as variables to the coordinates
    ds = ds.set_coords(Coordinates.intersection(ds.variables))
    ds = ds.rename_vars(Coordinates.remap(ds.coords))

    ds[LON], ds[LAT] = (ds[coord].compute() for coord in (LON, LAT))

    # - dimension assignment
    if missing_dims := Dimensions.difference(ds.dims):
        for dim in missing_dims:
            ds = ds.expand_dims(dim, axis=[DIMENSIONS.index(dim)])

    # # - coordinate assignment
    if missing_coords := Coordinates.difference(ds.coords):
        if missing_coords != {LVL}:
            raise ValueError(f"missing coordinates {missing_coords}; only {LVL} is allowed to be missing!")
        ds = ds.assign_coords(DERIVED_SURFACE_COORDINATE).set_xindex(LVL)

    if ds[LAT].dims == (Y,) and ds[LON].dims == (X,):
        # 5.2. Two-Dimensional Latitude, Longitude, Coordinate
        # Variables
        # The latitude and longitude coordinates of a horizontal grid that was not defined as a Cartesian
        # product of latitude and longitude axes, can sometimes be represented using two-dimensional
        # coordinate variables. These variables are identified as coordinates by use of the coordinates
        # attribute
        lon, lat = (ds[coord].to_numpy() for coord in (LON, LAT))
        yy, xx = np.meshgrid(lat, lon, indexing="ij")

        ds = ds.assign_coords({LAT: (LAT.axis, yy), LON: (LON.axis, xx)})

    ds = ds.transpose(*DIMENSIONS)

    return ds


# =====================================================================================================================
# - the dataset
# =====================================================================================================================
class IndependentDataset(xr.Dataset):
    __slots__ = ()

    @classmethod
    def from_dependant(cls, ds: xr.Dataset, **kwargs) -> Self:
        return cls(make_independent(ds), **kwargs)

    # - dims
    @property
    def t(self) -> xr.DataArray:
        return self[T]

    @property
    def z(self) -> xr.DataArray:
        return self[Z]

    @property
    def y(self) -> xr.DataArray:
        return self[Y]

    @property
    def x(self) -> xr.DataArray:
        return self[X]

    # - coords
    @property
    def time(self) -> xr.DataArray:
        return self[TIME]

    @property
    def level(self) -> xr.DataArray:
        return self[LVL]

    @property
    def lats(self) -> xr.DataArray:
        return self[LAT]

    @property
    def lons(self) -> xr.DataArray:
        return self[LON]


class DependentDataset(IndependentDataset):
    __slots__ = ()
    __dims__ = (T, Z, Y, X)
    __coords__ = (TIME, LVL, LAT, LON)

    def __init__(
        self, data: xr.Dataset, *, attrs: Mapping[str, Any] | None = None, depends: Depends | None = None
    ) -> None:
        # TODO:
        # - add method to create the dataset from a 4/5d array
        if isinstance(data, DependentDataset) and attrs is None:
            attrs = {
                _GRID_DEFINITION: data.grid_definition,
                _DEPENDS: data.depends,
            }
        super().__init__(data, attrs=attrs)
        if depends is not None:  # TODO: these 2 conditionals can be consolidated
            self.attrs[_DEPENDS] = depends

        if _GRID_DEFINITION not in self.attrs:
            lons, lats = (self[x].to_numpy() for x in (LON, LAT))
            lons = (lons + 180.0) % 360 - 180.0
            # TODO: need to write a test for this and insure
            # the lons are in the range of -180 to 180
            self.attrs[_GRID_DEFINITION] = GridDefinition(lons, lats)
        assert is_independent(self)

    @classmethod
    def from_zarr(cls, store: Any, depends: Depends) -> DependentDataset:
        depends = Dependencies(depends)
        return cls.from_dependant(xr.open_zarr(store, drop_variables=depends.difference), depends=depends)

    @property
    def grid_definition(self) -> GridDefinition:
        return self.attrs[_GRID_DEFINITION]

    @property
    def depends(self) -> Dependencies:
        return self.attrs[_DEPENDS]


# =====================================================================================================================
# - Resampling
# =====================================================================================================================
class Mesoscale(Data[NDArray[np.float_]]):
    def __init__(
        self,
        dx: float = 200.0,
        dy: float | None = None,
        *,
        rate: float = 1.0,
        pressure: ListLike[Number] = DEFAULT_PRESSURE,
        troposphere: ListLike[Number] | None = None,
    ) -> None:
        super().__init__()
        # - descending pressure
        tropo = np.asarray(
            sort_unique(self._arange() if troposphere is None else troposphere, descending=True), dtype=np.float_
        )
        self._hpa = hpa = np.asarray(sort_unique(pressure, descending=True), dtype=np.float_)
        if not all(np.isin(hpa, tropo)):
            raise ValueError(f"pressure {hpa} must be a subset of troposphere {tropo}")

        # - ascending scale
        mask = np.isin(tropo, hpa)
        self._scale = scale = log_scale(tropo, rate=rate)[::-1][mask]
        self._dx, self._dy = scale[np.newaxis] * np.array([[dx], [dy or dx]])

    @staticmethod
    def _arange(
        start: int = 1000,
        stop: int = 25 - 1,
        step: int = -25,
        *,
        p0: float = P0,
        p1=P1,
    ) -> Sequence[float]:
        return [p0, *range(start, stop, step), p1]

    @classmethod
    def arange(
        cls,
        dx: float = 200.0,
        dy: float | None = None,
        start: int = 1000,
        stop: int = 25 - 1,
        step: int = -25,
        *,
        p0: float = P0,
        p1=P1,
        rate: float = 1.0,
        pressure: ListLike[Number],
    ) -> Mesoscale:
        return cls(dx, dy, rate=rate, pressure=pressure, troposphere=cls._arange(start, stop, step, p0=p0, p1=p1))

    @property
    def hpa(self) -> NDArray[np.float_]:
        return self._hpa

    @property
    def scale(self) -> NDArray[np.float_]:
        return self._scale

    @property
    def dx(self) -> NDArray[np.float_]:
        return self._dx

    @property
    def dy(self) -> NDArray[np.float_]:
        return self._dy

    @property
    def data(self) -> Iterable[tuple[str, NDArray[np.float_]]]:
        yield from (("scale", self.scale), ("hpa", self.hpa), ("dx", self.dx), ("dy", self.dy))

    def __array__(self) -> Array[[N, N2], np.float_]:
        return self.to_numpy()

    def __len__(self) -> int:
        return len(self.hpa)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()).set_index("hpa").sort_index()

    def to_numpy(self, *, units: Unit = "km") -> Array[[N, N2], np.float_]:
        return np.c_[self.dx, self.dy] * _units[units]

    def stack_extent(self, *, units: Unit = "km") -> Array[[N, N4], np.float_]:
        xy = self.to_numpy(units=units)
        return np.c_[-xy, xy]

    def resample(
        self,
        *dsets: DependentDataset,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
        method: str = "nearest",
    ) -> ReSampler:
        return ReSampler(
            ReSampleInstructor(self, *dsets),
            height=height,
            width=width,
            target_projection=target_projection,
            method=method,
        )


# =====================================================================================================================
class ArrayProducer(DataWorker[TimeSlicePoint, Array[[N, N, N, N, N], np.float_]], AbstractInstructor):
    def __init__(
        self,
        indices: Iterable[TimeSlicePoint],
        *dsets: DependentDataset,
        scale: Mesoscale,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
    ) -> None:
        super().__init__(
            indices,
            hpa=scale.hpa,
            resampler=scale.resample(
                *dsets,
                height=height,
                width=width,
                target_projection=target_projection,
            ),
        )

    @property
    def sampler(self) -> ReSampler:
        return self.attrs["resampler"]

    @property
    def instructor(self) -> ReSampleInstructor:
        return self.sampler._instructor

    def __getitem__(self, idx: TimeSlicePoint) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        time, (lon, lat) = idx
        return self.sampler(lon, lat, time)

    def get_array(self, idx: TimeSlicePoint, /) -> xr.DataArray:
        data = self[idx]
        time, _ = idx

        return xr.DataArray(
            data,
            dims=(_VARIABLES, T, Z, Y, X),
            coords={
                _VARIABLES: self.dvars[0],
                LVL: (LVL.axis, self.levels),
                TIME: (TIME.axis, self.slice_time(time)),
            },
        )

    def get_dataset(self, idx: TimeSlicePoint, /) -> xr.Dataset:
        return self.get_array(idx).to_dataset(_VARIABLES)
