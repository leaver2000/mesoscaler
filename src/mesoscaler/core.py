from __future__ import annotations

import functools
import html

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import xarray.core.formatting_html
from pyresample.geometry import GridDefinition
from xarray.core.coordinates import DatasetCoordinates

from . import display
from ._typing import (
    N4,
    Any,
    Array,
    CanBeItems,
    Final,
    Hashable,
    Iterable,
    ListLike,
    Literal,
    Mapping,
    N,
    Nt,
    Number,
    Nv,
    Nx,
    Ny,
    Nz,
    PointOverTime,
    Self,
    Sequence,
    StrPath,
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
    DimensionsMapType,
    T,
    X,
    Y,
    Z,
)
from .generic import Data, DataWorker
from .sampling.domain import UNITS, AbstractDomain, DatasetSequence, Domain
from .sampling.resampler import ReSampler
from .utils import items, join_kv, log_scale, sort_unique

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
DEFAULT_RESAMPLE_METHOD: Final[Literal["nearest"]] = "nearest"

DERIVED_SURFACE_COORDINATE = {LVL: (LVL.axis, [STANDARD_SURFACE_PRESSURE])}
"""If the Dataset does not contain a vertical coordinate, it is assumed to be a derived atmospheric parameter
or near surface parameter. The vertical coordinate is then set to the standard surface pressure of 1013.25 hPa."""

_units: Mapping[Unit, float] = {"km": 1.0, "m": 1000.0}
_VARIABLES = "variables"
_GRID_DEFINITION_ATTRIBUTE = "grid_definition"
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
        return f"{self.__class__.__name__}({self.dataset_name})"

    @property
    def dataset_name(self) -> str:
        return self.enum.__name__

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


# =====================================================================================================================
# - Resampling
# =====================================================================================================================
class Mesoscale(Data[Array[[...], np.float_]]):
    def __init__(
        self,
        dx: float = DEFAULT_DX,
        dy: float | None = None,
        *,
        rate: float = DEFAULT_SCALE_RATE,
        levels: ListLike[Number] = DEFAULT_LEVELS,
        troposphere: ListLike[Number] | None = None,
        xy_units: str = "km",
        z_units: str = "hPa",
    ) -> None:
        if xy_units not in UNITS[X, Y]:
            raise ValueError(f"units must be one of {UNITS[X, Y]}")
        if z_units not in UNITS[Z]:
            raise ValueError(f"units must be one of {UNITS[Z]}")

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
        self._units: DimensionsMapType[str] = {(X, Y): xy_units, Z: z_units}

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

    #  - properties - #
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
    def unit(self) -> DimensionsMapType[str]:
        return self._units

    @property
    def data(self) -> Iterable[tuple[str, Array[[...], np.float_]]]:
        yield from (
            ("scale", self.scale),
            ("levels", self.levels),
            ("area_extent", self.area_extent),
        )  # type: ignore

    @property
    def plot(self) -> display.PlotAccessor:
        return display.PlotAccessor(self)

    @property
    def area_extent(self) -> Array[[N, N4], np.float_]:
        xy = np.c_[self.dx, self.dy]
        return np.c_[-xy, xy]

    #  - array methods - #
    def __array__(self) -> Array[[N, N4], np.float_]:
        return self.area_extent

    def to_numpy(self, *, units: Unit = "km") -> Array[[N, N4], np.float_]:
        return self.area_extent * _units[units]

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(
            np.asarray(self), columns=["dx0", "dy0", "dx1", "dy1"], index=pd.Index(self.levels, name="pressure")
        )
        df.insert(0, "scale", self.scale)
        return df.sort_index()

    def __repr__(self) -> str:
        name = self.name
        size = self.size
        xy, z = self.unit.values()
        return join_kv(
            f"{name}({size=}):",
            ("scale", self.scale),
            (f"levels[{z}]", self.levels),
            (f"extent[{xy}]", self.area_extent),
        )

    def __len__(self) -> int:
        return len(self.levels)

    # - methods - #
    def get_domain(self, dsets: Iterable[DependentDataset]) -> Domain:
        return Domain(dsets, self)

    def resample(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *,
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
    ) -> ReSampler:
        return ReSampler(
            __x if isinstance(__x, Domain) else self.get_domain(__x), height=height, width=width, method=method
        )


# =====================================================================================================================
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

    def select_from(self, x: Mapping[Coordinates, Sequence[Any]]) -> Self:
        return self.sel({k: self[k].isin(v) for k, v in x.items()})

    @property
    def grid_definition(self) -> GridDefinition:
        if _GRID_DEFINITION_ATTRIBUTE not in self.attrs:
            self.attrs[_GRID_DEFINITION_ATTRIBUTE] = self.get_grid_definition()
        return self.attrs[_GRID_DEFINITION_ATTRIBUTE]

    def get_grid_definition(self, nprocs: int = 1) -> GridDefinition:
        # TODO: need to write a test for this and insure
        # the lons are in the range of -180 to 180
        lons, lats = (self[x].to_numpy() for x in (LON, LAT))
        lons = (lons + 180.0) % 360 - 180.0
        return GridDefinition(lons, lats, nprocs=nprocs)

    def set_grid_definition(self, grid_definition: GridDefinition | None = None) -> Self:
        self.attrs[_GRID_DEFINITION_ATTRIBUTE] = grid_definition or self.get_grid_definition()
        return self


class DependentDataset(IndependentDataset):
    __slots__ = ()

    def __init__(
        self,
        data: xr.Dataset | Mapping[DependentVariables, xr.DataArray],
        coords: Mapping[Any, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        *,
        depends: Depends | None = None,
    ) -> None:
        # TODO:
        # - add method to create the dataset from a 4/5d array
        if isinstance(data, DependentDataset) and attrs is None:
            attrs = {
                _DEPENDS: data.depends,
                # we dont want to forward the grid definition
                # because the dataset may have been sliced or something
                # so we can't forward the definition
            }
        elif attrs is None and depends is not None:
            attrs = {_DEPENDS: Dependencies(depends)}

        super().__init__(data, coords, attrs)

    @classmethod
    def from_zarr(cls, store: Any, depends: Depends) -> DependentDataset:
        depends = Dependencies(depends)
        return cls.from_dependant(xr.open_zarr(store, drop_variables=depends.difference), depends=depends)

    @property
    def depends(self) -> Dependencies:
        return self.attrs[_DEPENDS]

    def _repr_html_(self) -> str:
        obj_type = f"{self.__class__.__name__}[{self.depends.dataset_name}]"

        header_components = [f"<div class='xr-obj-type'>{html.escape(obj_type)}</div>"]

        sections = [
            xarray.core.formatting_html.dim_section(self),
            xarray.core.formatting_html.coord_section(self.coords),
            xarray.core.formatting_html.datavar_section(self.data_vars),
        ]

        return xarray.core.formatting_html._obj_repr(self, header_components, sections)


# =====================================================================================================================
#
# =====================================================================================================================
class DataProducer(DataWorker[PointOverTime, Array[[Nv, Nt, Nz, Ny, Nx], np.float_]], AbstractDomain):
    def __init__(self, indices: Iterable[PointOverTime], resampler: ReSampler) -> None:
        super().__init__(indices, resampler=resampler)

    @functools.cached_property
    def sampler(self) -> ReSampler:
        return self.attrs["resampler"]

    @property
    def domain(self) -> Domain:
        return self.sampler.domain

    def __getitem__(self, idx: PointOverTime) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
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
                # TIME: (TIME.axis, self.slice_time(time)),
            },
        )

    def get_dataset(self, idx: PointOverTime, /) -> xr.Dataset:
        return self.get_array(idx).to_dataset(_VARIABLES)


# =====================================================================================================================
#  - functions
# =====================================================================================================================
def _open_datasets(
    paths: CanBeItems[StrPath, Depends], *, levels: ListLike[Number] | None = None, times: Any = None
) -> Iterable[DependentDataset]:
    for path, depends in items(paths):
        ds = DependentDataset.from_zarr(path, depends)
        if levels is not None and times is not None:
            ds = ds.sel({LVL: ds.level.isin(levels), TIME: ds.time.isin(times)})
        elif levels is not None:
            ds = ds.sel({LVL: ds.level.isin(levels)})
        elif times is not None:
            ds = ds.sel({TIME: ds.time.isin(times)})

        yield ds


def open_datasets(
    paths: CanBeItems[StrPath, Depends], *, levels: ListLike[Number] | None = None, times: Any = None
) -> DatasetSequence:
    return DatasetSequence(_open_datasets(paths, levels=levels, times=times))
