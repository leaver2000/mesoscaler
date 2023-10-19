from __future__ import annotations

import functools

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
    Callable,
    Final,
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
from .generic import Data, DataGenerator, DataWorker
from .sampling.intersection import AbstractIntersection, DatasetIntersection
from .sampling.resampler import Nt, Nv, Nx, Ny, Nz, ReSampler
from .sampling.sampler import LinearSampler
from .utils import log_scale, sort_unique

Depends: TypeAlias = Union[type[DependentVariables], DependentVariables, Sequence[DependentVariables], "Dependencies"]

Unit = Literal["km", "m"]
# =====================================================================================================================
# - hPa scaling -
DEFAULT_PRESSURE_BASE = STANDARD_SURFACE_PRESSURE = P0 = 1013.25  # - hPa
DEFAULT_PRESSURE_TOP = P1 = 25.0  # - hPa

DEFAULT_LEVELS: ListLike[Number] = [P0, 925.0, 850.0, 700.0, 500.0, 300.0]  # - hPa

DEFAULT_LEVEL_START = 1000  # - hPa
DEFAULT_LEVEL_STOP = 25 - 1
DEFAULT_LEVEL_STEP = -25
# -
DEFAULT_HEIGHT = DEFAULT_WIDTH = 80  # - px
DEFAULT_DX = DEFAULT_DY = 200.0  # - km
DEFAULT_SCALE_RATE = 15.0
DEFAULT_TARGET_PROJECTION: Final[LiteralCRS] = "lambert_azimuthal_equal_area"
DEFAULT_RESAMPLE_METHOD: Final[Literal["nearest"]] = "nearest"

DERIVED_SURFACE_COORDINATE = {LVL: (LVL.axis, [STANDARD_SURFACE_PRESSURE])}
"""If the Dataset does not contain a vertical coordinate, it is assumed to be a derived atmospheric parameter
or near surface parameter. The vertical coordinate is then set to the standard surface pressure of 1013.25 hPa."""

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

    # @property
    # def name(self) -> str:
    #     return self.enum.name


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
        dx: float = DEFAULT_DX,
        dy: float | None = None,
        *,
        rate: float = DEFAULT_SCALE_RATE,
        levels: ListLike[Number] = DEFAULT_LEVELS,
        troposphere: ListLike[Number] | None = None,
    ) -> None:
        super().__init__()
        # - descending pressure
        tropo = np.asarray(
            sort_unique(self._arange() if troposphere is None else troposphere, descending=True), dtype=np.float_
        )
        self._levels = lvls = np.asarray(sort_unique(levels, descending=True), dtype=np.float_)
        if not all(np.isin(lvls, tropo)):
            raise ValueError(f"pressure {lvls} must be a subset of troposphere {tropo}")

        # - ascending scale
        mask = np.isin(tropo, lvls)
        self._scale = scale = log_scale(tropo, rate=rate)[::-1][mask]
        self._dx, self._dy = scale[np.newaxis] * np.array([[dx], [dy or dx]])

    @staticmethod
    def _arange(
        start: int = DEFAULT_LEVEL_START,
        stop: int = DEFAULT_LEVEL_STOP,
        step: int = DEFAULT_LEVEL_STEP,
        *,
        p0: float = DEFAULT_PRESSURE_BASE,
        p1: float = DEFAULT_PRESSURE_TOP,
    ) -> Sequence[float]:
        return [p0, *range(start, stop, step), p1]

    @classmethod
    def arange(
        cls,
        dx: float = DEFAULT_DX,
        dy: float | None = None,
        start: int = DEFAULT_LEVEL_START,
        stop: int = DEFAULT_LEVEL_STOP,
        step: int = DEFAULT_LEVEL_STEP,
        *,
        p0: float = DEFAULT_PRESSURE_BASE,
        p1: float = DEFAULT_PRESSURE_TOP,
        rate: float = DEFAULT_SCALE_RATE,
        levels: ListLike[Number] = DEFAULT_LEVELS,
    ) -> Mesoscale:
        return cls(dx, dy, rate=rate, levels=levels, troposphere=cls._arange(start, stop, step, p0=p0, p1=p1))

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self._levels

    @property
    def scale(self) -> Array[[N], np.float_]:
        return self._scale

    @property
    def dx(self) -> Array[[N], np.float_]:
        return self._dx

    @property
    def dy(self) -> Array[[N], np.float_]:
        return self._dy

    @property
    def area_extent(self) -> Array[[N, N4], np.float_]:
        xy = np.c_[self.dx, self.dy]
        return np.c_[-xy, xy]

    @property
    def data(self) -> Iterable[tuple[str, NDArray[np.float_]]]:
        yield from (("scale", self.scale), ("levels", self.levels), ("dx", self.dx), ("dy", self.dy))

    def __array__(self) -> Array[[N, N2], np.float_]:
        return self.to_numpy()

    def __len__(self) -> int:
        return len(self.levels)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()).set_index("levels").sort_index()

    def to_numpy(self, *, units: Unit = "km") -> Array[[N, N2], np.float_]:
        xy = np.c_[self.dx, self.dy] * _units[units]
        return np.c_[-xy, xy]

    def intersection(self, *dsets: DependentDataset) -> DatasetIntersection:
        return DatasetIntersection.from_datasets(dsets, self)

    def resample(
        self,
        *dsets: DependentDataset,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
        method: str = "nearest",
    ) -> ReSampler:
        return ReSampler(
            self.intersection(*dsets), height=height, width=width, target_projection=target_projection, method=method
        )


# =====================================================================================================================
#
# =====================================================================================================================
class DataProducer(DataWorker[TimeSlicePoint, Array[[N, N, N, N, N], np.float_]], AbstractIntersection):
    def __init__(self, indices: Iterable[TimeSlicePoint], resampler: ReSampler) -> None:
        super().__init__(indices, resampler=resampler)

    @functools.cached_property
    def sampler(self) -> ReSampler:
        return self.attrs["resampler"]

    @property
    def intersection(self) -> DatasetIntersection:
        return self.sampler.intersection

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


# =====================================================================================================================
#  - functions
# =====================================================================================================================
def open_datasets(
    paths: Iterable[tuple[str, Depends]], *, levels: ListLike[Number] | None = None
) -> Iterable[DependentDataset]:
    for path, depends in paths:
        ds = DependentDataset.from_zarr(path, depends)
        if levels is not None:
            ds = ds.sel({LVL: ds.level.isin(levels)})
        yield ds


def create_resampler(
    dsets: Iterable[DependentDataset],
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    *,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = DEFAULT_SCALE_RATE,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    target_projection: LiteralCRS = DEFAULT_TARGET_PROJECTION,
    method: str = DEFAULT_RESAMPLE_METHOD,
) -> ReSampler:
    return Mesoscale.arange(
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
    ).resample(*dsets, height=height, width=width, target_projection=target_projection, method=method)


def data_producer(
    dsets: Iterable[DependentDataset],
    indices: Iterable[TimeSlicePoint] | Callable[[DatasetIntersection], Iterable[TimeSlicePoint]] = LinearSampler,
    *,
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = DEFAULT_SCALE_RATE,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    target_projection: LiteralCRS = DEFAULT_TARGET_PROJECTION,
    method: str = DEFAULT_RESAMPLE_METHOD,
    **sampler_kwargs: Any,
) -> DataProducer:
    scale = Mesoscale.arange(
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
    )
    resampler = scale.resample(*dsets, height=height, width=width, target_projection=target_projection, method=method)
    if callable(indices):
        indices = indices(resampler.intersection, **sampler_kwargs)

    return DataProducer(indices, resampler=resampler)


def data_generator(
    paths: Iterable[tuple[str, Depends]],
    indices: Iterable[TimeSlicePoint] | Callable[[DatasetIntersection], Iterable[TimeSlicePoint]] = LinearSampler,
    *,
    dx: float = 200,
    dy: float | None = None,
    start: int = 1000,
    stop: int = 25 - 1,
    step: int = -25,
    p0: float = P0,
    p1: float = P1,
    rate: float = 1,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = 80,
    width: int = 80,
    target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
    method: str = "nearest",
    # - data consumer -
    maxsize: int = 0,
    timeout: float | None = None,
    **sampler_kwargs: Any,
) -> DataGenerator[Array[[N, N, N, N, N], np.float_]]:
    datasets = open_datasets(paths, levels=levels)
    producer = data_producer(
        datasets,
        indices,
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
        height=height,
        width=width,
        target_projection=target_projection,
        method=method,
        **sampler_kwargs,
    )
    return DataGenerator(producer, maxsize=maxsize, timeout=timeout)


from . import _compat


def data_loader(
    paths: Iterable[tuple[str, Depends]],
    indices: Iterable[TimeSlicePoint] | Callable[[DatasetIntersection], Iterable[TimeSlicePoint]] = LinearSampler,
    *,
    dx: float = 200,
    dy: float | None = None,
    start: int = 1000,
    stop: int = 25 - 1,
    step: int = -25,
    p0: float = P0,
    p1: float = P1,
    rate: float = 1,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = 80,
    width: int = 80,
    target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
    method: str = "nearest",
    # - data consumer -
    maxsize: int = 0,
    timeout: float | None = None,
    #
    batch_size: int | None = 1,
    shuffle: bool | None = None,
    # sampler: Sampler[Unknown] | Iterable[Unknown] | None = None,
    # batch_sampler: Sampler[List[Unknown]] | Iterable[List[Unknown]] | None = None,
    num_workers: int = 0,
    # collate_fn: _collate_fn_t[Unknown] | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    # timeout: float = 0,
    **sampler_kwargs: Any,
):
    if not _compat._has_torch:
        raise RuntimeError("torch is not installed!")
    dataset = data_generator(
        paths,
        indices,
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
        height=height,
        width=width,
        target_projection=target_projection,
        method=method,
        maxsize=maxsize,
        timeout=timeout,
        **sampler_kwargs,
    )
    return _compat.DataLoader(
        dataset,
        batch_size=batch_size,
        timeout=timeout or 0,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
