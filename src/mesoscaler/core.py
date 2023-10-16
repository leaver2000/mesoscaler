from __future__ import annotations

import abc
import functools

import numpy as np
import pandas as pd
import pyproj
import pyresample.geometry
import xarray as xr
from pyresample.geometry import AreaDefinition, GridDefinition
from xarray.core.coordinates import DatasetCoordinates

from ._typing import (
    N2,
    N4,
    Any,
    AreaExtent,
    Array,
    Hashable,
    Iterable,
    Iterator,
    Latitude,
    ListLike,
    Literal,
    Longitude,
    Mapping,
    N,
    NDArray,
    Number,
    PointOverTime,
    Self,
    Sequence,
    Slice,
    TimeSlice,
    TypeAlias,
    Union,
)
from .enums import (
    DIMENSIONS,
    LAT,
    LON,
    LVL,
    TIME,
    CoordinateReferenceSystem,
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
from .utils import area_definition, log_scale, slice_time, sort_unique

Depends: TypeAlias = Union[type[DependentVariables], DependentVariables, Sequence[DependentVariables], "Dependencies"]
ResampleInstruction: TypeAlias = tuple["DependentDataset", AreaExtent]
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
    ) -> ListLike[Number]:
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
            _Instruction(self, *dsets),
            height=height,
            width=width,
            target_projection=target_projection,
            method=method,
        )


# =====================================================================================================================
#
# =====================================================================================================================
class AbstractInstruction(abc.ABC):
    @property
    @abc.abstractmethod
    def instruction(self) -> _Instruction:
        ...

    @property
    def dvars(self) -> Array[[N, N], np.str_]:
        return self.instruction._dvars

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self.instruction._levels

    @property
    def time(self) -> Array[[N], np.datetime64]:
        return self.instruction._time

    def slice_time(self, s: TimeSlice, /) -> Array[[N], np.datetime64]:
        return slice_time(self.time, s)


class _Instruction(Iterable[ResampleInstruction], AbstractInstruction):
    def __init__(self, scale: Mesoscale, *dsets: DependentDataset) -> None:
        super().__init__()

        time, levels, dvars = zip(*((ds.time.to_numpy(), ds.level.to_numpy(), list(ds.data_vars)) for ds in dsets))
        # - time
        self._time = time = np.sort(np.unique(time))

        # - levels
        levels = np.concatenate(levels)
        self._levels = levels = np.sort(levels[np.isin(levels, scale.hpa)])[::-1]  # descending

        # - dvars
        self._dvars = np.stack(dvars)

        datasets = (ds.sel({LVL: [lvl], TIME: time}) for lvl in levels for ds in dsets if lvl in ds.level)
        extents = scale.stack_extent() * 1000.0  # km -> m
        self._it = tuple(zip(datasets, extents))

    def __iter__(self) -> Iterator[ResampleInstruction]:
        return iter(self._it)

    @property
    def instruction(self) -> _Instruction:
        return self


def _get_partial_method(
    method,
    sigmas=[1.0],
    radius_of_influence=500000,
    fill_value=0,
    reduce_data=True,
    nprocs=1,
    segments=None,
    with_uncert: bool = False,
) -> functools.partial[NDArray[np.float_]]:
    if method == "nearest":
        func = pyresample.kd_tree.resample_nearest
        kwargs = dict(
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
            reduce_data=reduce_data,
            nprocs=nprocs,
            segments=segments,
        )
    elif method == "gauss":
        func = pyresample.kd_tree.resample_gauss
        kwargs = dict(
            sigmas=sigmas,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
            reduce_data=reduce_data,
            nprocs=nprocs,
            segments=segments,
            with_uncert=with_uncert,
        )
    else:
        raise ValueError(f"method {method} is not supported!")
    return functools.partial(func, **kwargs)


class ReSampler(AbstractInstruction):
    def __init__(
        self,
        instruction: _Instruction,
        /,
        *,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
        method: str = "nearest",
        sigmas=[1.0],
        radius_of_influence=500000,
        fill_value=0,
        reduce_data=True,
        nprocs=1,
        segments=None,
        with_uncert: bool = False,
    ) -> None:
        self._instruction = instruction
        self.height = height
        self.width = width
        self.target_projection = (
            CoordinateReferenceSystem[target_projection] if isinstance(target_projection, str) else target_projection
        )

        self._resample_method = _get_partial_method(
            method,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
            reduce_data=reduce_data,
            nprocs=nprocs,
            segments=segments,
            with_uncert=with_uncert,
            sigmas=sigmas,
        )

    @property
    def instruction(self) -> _Instruction:
        return self._instruction

    def _partial_area_definition(self, longitude: Longitude, latitude: Latitude) -> functools.partial[AreaDefinition]:
        return functools.partial(
            area_definition,
            width=self.width,
            height=self.height,
            projection=self.target_projection.from_point(longitude, latitude),
        )

    def _resample_point_over_time(
        self, longitude: Longitude, latitude: Latitude, time: Slice[np.datetime64]
    ) -> list[NDArray]:
        area_definition = self._partial_area_definition(longitude, latitude)

        return [
            self._resample_method(
                ds.grid_definition,
                ds.sel({TIME: time}).to_stacked_array("C", [Y, X]).to_numpy(),
                area_definition(area_extent=area_extent),
            )
            for ds, area_extent in self._instruction
        ]

    def __call__(self, longitude: Longitude, latitude: Latitude, time: TimeSlice) -> Array[[N, N, N, N, N], np.float_]:
        arr = np.stack(self._resample_point_over_time(longitude, latitude, time))
        # - reshape the data
        t = len(self.slice_time(time))  # (time.stop - time.start) // np.timedelta64(1, "h") + 1

        z, y, x = arr.shape[:3]
        arr = arr.reshape((z, y, x, t, -1))  # unsqueeze C
        return np.moveaxis(arr, (-1, -2), (0, 1))


# =====================================================================================================================
class ArrayWorker(DataWorker[PointOverTime, Array[[N, N, N, N, N], np.float_]], AbstractInstruction):
    def __init__(
        self,
        indices: Iterable[PointOverTime],
        *dsets: DependentDataset,
        scale: Mesoscale,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
    ) -> None:
        super().__init__(
            indices,
            hpa=scale.hpa,
            sampler=scale.resample(
                *dsets,
                height=height,
                width=width,
                target_projection=target_projection,
            ),
        )

    @property
    def sampler(self) -> ReSampler:
        return self.attrs["sampler"]

    @property
    def instruction(self) -> _Instruction:
        return self.sampler._instruction

    def __getitem__(self, idx: PointOverTime) -> Array[[N, N, N, N, N], np.float_]:
        (lon, lat), time = idx
        return self.sampler(lon, lat, time)

    def get_array(self, idx: PointOverTime, /) -> xr.DataArray:
        data = self[idx]
        _, time = idx

        return xr.DataArray(
            data,
            dims=(_VARIABLES, T, Z, Y, X),
            coords={
                _VARIABLES: self.dvars[0],
                LVL: (LVL.axis, self.levels),
                TIME: (TIME.axis, self.slice_time(time)),
            },
        )

    def get_dataset(self, idx: PointOverTime, /) -> xr.Dataset:
        return self.get_array(idx).to_dataset(_VARIABLES)
