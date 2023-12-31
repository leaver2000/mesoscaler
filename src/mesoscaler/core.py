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

from . import _compat as compat, utils
from ._typing import (
    N4,
    Any,
    Array,
    ChainableItems,
    Final,
    Hashable,
    Iterable,
    ListLike,
    Literal,
    Mapping,
    N,
    NamedTuple,
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
    overload,
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
    T,
    X,
    Y,
    Z,
)
from .generic import Data, DataWorker
from .sampling.domain import DatasetSequence, Domain
from .sampling.resampler import (
    AbstractResampler,
    ReSampler,
    ResamplingPipeline,
    SamplerT,
)
from .sampling.sampler import (
    AreaOfInterestSampler,
    MultiPointSampler,
    PointOverTimeSampler,
)

Depends: TypeAlias = Union[type[DependentVariables], DependentVariables, Sequence[DependentVariables], "Dependencies"]

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
Unit = Literal["km", "m"]


class Units(NamedTuple):
    xy: Unit
    z: str


class Mesoscale(Data[Array[[...], np.float_]]):
    def __init__(
        self,
        dx: float = DEFAULT_DX,
        dy: float | None = None,
        *,
        rate: float = DEFAULT_SCALE_RATE,
        levels: ListLike[Number] = DEFAULT_LEVELS,
        troposphere: ListLike[Number] | None = None,
        xy_units: Unit = "km",
        z_units: str = "hPa",
    ) -> None:
        super().__init__()
        # - descending pressure
        tropo = np.asarray(
            utils.sort_unique(self._arange() if troposphere is None else troposphere, descending=True), dtype=np.float_
        )
        self._levels = lvls = np.asarray(utils.sort_unique(levels, descending=True), dtype=np.float_)
        if not all(np.isin(lvls, tropo)):
            raise ValueError(f"pressure {lvls} must be a subset of troposphere {tropo}")

        # - ascending scale
        mask = np.isin(tropo, lvls)
        self._scale = scale = utils.log_scale(tropo, rate=rate)[::-1][mask]
        self._dx, self._dy = scale[np.newaxis] * np.array([[dx], [dy or dx]])
        self._units = Units(xy=xy_units, z=z_units)

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

    #  - properties
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
    def units(self) -> Units:
        return self._units

    @property
    def data(self) -> Iterable[tuple[str, Array[[...], np.float_]]]:
        yield from (
            ("scale", self.scale),
            ("levels", self.levels),
            ("area_extent", self.area_extent),
        )  # type: ignore

    @property
    def area_extent(self) -> Array[[N, N4], np.float_]:
        xy = np.c_[self.dx, self.dy]
        return np.c_[-xy, xy]

    #  - dunder methods
    def __repr__(self) -> str:
        name = self.name
        size = self.size
        xy, z = self.units
        return utils.join_kv(
            f"{name}({size=}):",
            ("scale", self.scale),
            (f"levels[{z}]", self.levels),
            (f"extent[{xy}]", self.area_extent),
        )

    def __len__(self) -> int:
        return len(self.levels)

    def __array__(self) -> Array[[N, N4], np.float_]:
        return self.area_extent

    # - methods
    def to_numpy(self, *, units: Unit = "km") -> Array[[N, N4], np.float_]:
        u = self.units.xy
        x = self.area_extent
        if u == units:
            return x
        elif u == "km":
            return x * 1000.0
        elif u == "m":
            return x / 1000.0

        raise ValueError(f"units {u} is not supported!")

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(
            np.asarray(self), columns=["dx0", "dy0", "dx1", "dy1"], index=pd.Index(self.levels, name="pressure")
        )
        df.insert(0, "scale", self.scale)
        return df.sort_index()

    def get_domain(self, dsets: Iterable[DependentDataset]) -> Domain:
        return Domain(dsets, self)

    def create_resampler(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *,
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
    ) -> ReSampler:
        domain = __x if isinstance(__x, Domain) else self.get_domain(__x)
        return ReSampler(domain, height=height, width=width, method=method)

    @overload
    def create_pipeline(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *points: tuple[float, float],
        height: int = ...,
        width: int = ...,
        method: str = ...,
        time_step: int = ...,
    ) -> ResamplingPipeline[MultiPointSampler]:
        ...

    @overload
    def create_pipeline(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *,
        sampler: SamplerT,
        height: int = ...,
        width: int = ...,
        method: str = ...,
    ) -> ResamplingPipeline[SamplerT]:
        ...

    @overload
    def create_pipeline(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *,
        height: int = ...,
        width: int = ...,
        method: str = ...,
        aoi: tuple[float, float, float, float] | None = ...,
        lon_lat_step: int = ...,
        padding: tuple[float, float] | float | None = ...,
    ) -> ResamplingPipeline[AreaOfInterestSampler]:
        ...

    def create_pipeline(
        self,
        __x: Domain | Iterable[DependentDataset],
        /,
        *points: tuple[float, float],
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
        time_step: int = 1,
        sampler: PointOverTimeSampler | None = None,
        aoi: tuple[float, float, float, float] | None = None,
        lon_lat_step: int = 5,
        padding: tuple[float, float] | float | None = 2.5,
    ) -> ResamplingPipeline[Any]:
        resampler = self.create_resampler(__x, height=height, width=width, method=method)
        return resampler.create_pipeline(
            sampler, *points, aoi=aoi, lon_lat_step=lon_lat_step, time_step=time_step, padding=padding
        )

    def produce(
        self,
        __x: Domain | Iterable[DependentDataset],
        indices: Iterable[PointOverTime],
        /,
        *,
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
    ):
        resampler = self.create_resampler(__x, height=height, width=width, method=method)
        return DataProducer(indices, resampler=resampler)

    def show(
        self,
        figsize: tuple[float, float] = (15, 6),
        *,
        linestyle="-",
        linewidth: float = 0.75,
        marker: str = ".",
        markersize: float = 2.5,
    ) -> None:
        X1, X2, Y = self.dx, self.dy, self.levels
        fig = compat.plt.figure(figsize=figsize)
        ax = fig.add_subplot(121)
        ax.invert_yaxis()
        ax.plot(X1, Y, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize)
        ax.plot(X2, Y, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_xlabel("Extent (km)")
        ax.legend(["dx", "dy"])

        df = self.to_pandas()
        ax2 = fig.add_subplot(122)
        ax2.axis("off")
        ax2.table(
            cellText=df.to_numpy().round(2).astype(str).tolist(),
            rowLabels=df.index.tolist(),
            bbox=[0, 0, 1, 1],  # type: ignore
            colLabels=df.columns.tolist(),
        )


# =====================================================================================================================
class DataProducer(DataWorker[PointOverTime, Array[[Nv, Nt, Nz, Ny, Nx], np.float_]], AbstractResampler):
    def __init__(self, indices: Iterable[PointOverTime], resampler: ReSampler) -> None:
        super().__init__(indices, resampler=resampler)

    # - AbstractResampler -
    @functools.cached_property
    def resampler(self) -> ReSampler:
        return self.attrs["resampler"]

    # - Mapping Interface -
    def __getitem__(self, idx: PointOverTime) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        (lon, lat), time = idx
        return self(lon, lat, time)


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
#  - functions
# =====================================================================================================================
def _open_datasets(
    paths: ChainableItems[StrPath, Depends], *, levels: ListLike[Number] | None = None, times: Any = None
) -> Iterable[DependentDataset]:
    for path, depends in utils.chain_items(paths):
        ds = DependentDataset.from_zarr(path, depends)
        if levels is not None and times is not None:
            ds = ds.sel({LVL: ds.level.isin(levels), TIME: ds.time.isin(times)})
        elif levels is not None:
            ds = ds.sel({LVL: ds.level.isin(levels)})
        elif times is not None:
            ds = ds.sel({TIME: ds.time.isin(times)})

        yield ds


def open_datasets(
    paths: ChainableItems[StrPath, Depends], *, levels: ListLike[Number] | None = None, times: Any = None
) -> DatasetSequence:
    return DatasetSequence(_open_datasets(paths, levels=levels, times=times))
