# @dataclasses.dataclass(frozen=True)
from __future__ import annotations

import abc
import functools
import textwrap
from typing import Iterable

import numpy as np

from .. import utils
from .._typing import (
    N4,
    TYPE_CHECKING,
    Any,
    AreaExtent,
    Array,
    Iterable,
    Iterator,
    N,
    NamedTuple,
    Self,
    Sequence,
    TimeSlice,
)

# from pyresample.geometry import AreaDefinition, GridDefinition
from ..enums import LAT, LON, LVL, TIME, Coordinates, DimensionsMapType, X, Y, Z
from ..generic import Callable, DataSampler, DataSequence, Mapping, TypeVar
from ..utils import repr_, slice_time

_T = TypeVar("_T")

if TYPE_CHECKING:
    from ..core import DependentDataset, Mesoscale
else:
    Mesoscale = Any
    DependentDataset = Any


UNITS: DimensionsMapType[tuple[str, ...]] = {(X, Y): ("km", "m"), Z: ("hPa",)}
DatasetAndExtent = tuple[DependentDataset, AreaExtent]


class BoundingBox(NamedTuple):
    """
    >>> min_lon, min_lat, max_lon, max_lat = -180.0, 180.0, -90.0, 90.0
    """

    west: float
    south: float
    east: float
    north: float

    def intersects(self, other: BoundingBox) -> bool:
        return (
            self.west < other.east and other.west < self.east and self.south < other.north and other.south < self.north
        )

    def contains(self, other: BoundingBox) -> bool:
        return (
            other.west >= self.west
            and other.east <= self.east
            and other.south >= self.south
            and other.north <= self.north
        )

    def __array__(self) -> Array[[N4], np.float_]:
        return np.array(self)

    def to_numpy(self) -> Array[[N4], np.float_]:
        return np.array(self)


class DatasetSequence(DataSequence[DependentDataset]):
    def get_coordinates(self) -> dict[Coordinates, list[Array[[...], Any]]]:
        return {
            LON: [((ds[LON].to_numpy() - 180.0) % 360 - 180.0) for ds in self],
            LAT: [ds[LAT].to_numpy() for ds in self],
            TIME: [ds[TIME].to_numpy() for ds in self],
            LVL: [ds[LVL].to_numpy() for ds in self],
        }

    def get_intersection(self, scale: Mesoscale) -> DomainIntersection:
        return DomainIntersection(self, scale)

    def fit_lon_lat(self, bbox: BoundingBox):
        return DatasetSequence(_fit_lon_lat(self, bbox))

    def batch_time(self, batcher: Array[[N, N], np.datetime64]):
        return DatasetSequence(_batch_time(self, batcher))

    def _repr_html_(self) -> str:
        return "\n".join(ds._repr_html_() for ds in self)

    def fit_domain(self, domain: AbstractDomain) -> DatasetSequence:
        min_lon, min_lat, max_lon, max_lat = domain.bbox
        min_lon = (min_lon - 180.0) % 360 + 180.0
        max_lon = (max_lon - 180.0) % 360 + 180.0

        return DatasetSequence(
            ds.sel(
                {
                    LVL: ds[LVL].isin(domain.levels),
                    TIME: ds[TIME].isin(domain.time),
                    X: ((ds[LON] >= min_lon) & (ds[LON] <= max_lon)).any(axis=0),
                    Y: ((ds[LAT] >= min_lat) & (ds[LAT] <= max_lat)).any(axis=1),
                }
            ).set_grid_definition()
            for ds in self
        )

    def batch_from(self, x: Mapping[Coordinates, Iterable[Sequence[float | np.datetime64]]]):
        return DatasetSequence(
            ds.select_from({k: batch}) for ds in self for k, batcher in x.items() for batch in batcher
        )


class AbstractDomain(abc.ABC):
    @property
    @abc.abstractmethod
    def domain(self) -> DomainIntersection:
        ...

    @property
    def bbox(self) -> BoundingBox:
        return self.domain._bbox

    @property
    def min_lon(self) -> float:
        return self.bbox.west

    @property
    def max_lon(self) -> float:
        return self.bbox.east

    @property
    def min_lat(self) -> float:
        return self.bbox.south

    @property
    def max_lat(self) -> float:
        return self.bbox.north

    # - AreaExtent -
    @property
    def area_extents(self) -> Array[[N, N4], np.float_]:
        return self.domain._area_extents

    #  - TimeSlice -
    @property
    def time_slice(self) -> TimeSlice:
        return self.domain._time_slice

    @property
    def min_time(self) -> np.datetime64:
        return self.time_slice.start

    @property
    def max_time(self) -> np.datetime64:
        return self.time_slice.stop

    @property
    def time(self) -> Array[[N], np.datetime64]:
        return self.domain._time

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self.domain._levels

    @property
    def dvars(self) -> Array[[N, N], np.str_]:
        return self.domain._dvars

    @property
    def n_vars(self) -> int:
        return self.dvars.shape[-1]

    @property
    def datasets(self) -> DatasetSequence:
        return self.domain._datasets

    def get_min_max(self, key: str) -> tuple[Any, Any]:
        if key == "lon":
            return self.min_lon, self.max_lon
        elif key == "lat":
            return self.min_lat, self.max_lat
        elif key == "time":
            return self.min_time, self.max_time
        else:
            raise KeyError(f"key {key} is not supported!")

    def linspace(self, key: str, *, frequency: int, round: int = 5) -> Array[[N], Any]:
        assert isinstance(frequency, int), 'frequency must be an integer when key is "lon" or "lat"'
        mn, mx = self.get_min_max(key)
        return np.linspace(mn, mx, frequency).round(round)

    def slice_time(self, s: TimeSlice, /) -> Array[[N], np.datetime64]:
        return slice_time(self.time, s)

    def iter_dataset_and_extent(self) -> Iterator[DatasetAndExtent]:
        return zip(self.datasets, self.area_extents)

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.n_vars,
            len(self.time),
            len(self.levels),
        )

    def batch_time(self, n: int) -> Array[[N, N], np.datetime64]:
        return utils.batch(self.time, n, strict=True)

    def _repr_args(self):
        return (
            ("bbox", repr_(self.bbox)),
            ("time", repr_(self.time)),
            ("scale", "\n" + textwrap.indent(repr_(self.scale), "  ")),
        )

    @property
    def scale(self) -> Mesoscale:
        return self.domain._scale

    def get_extents(self, units: str = "m") -> Array[[N, N4], np.float_]:
        if units not in UNITS[X, Y]:
            raise ValueError(f"units must be one of {UNITS[X, Y]}")

        extents = self.area_extents
        if units != self.scale.unit[X, Y]:
            if units == "km":
                extents /= 1000.0
            elif units == "m":
                extents *= 1000.0

        return extents


class DomainIntersectionSampler(
    DataSampler[_T],
    AbstractDomain,
    abc.ABC,
):
    @property
    def domain(self) -> DomainIntersection:
        return self._domain

    def __init__(self, domain: DomainIntersection) -> None:
        super().__init__()
        self._domain = domain

    @classmethod
    def partial(cls, *args: Any, **kwargs: Any) -> Callable[[DomainIntersection], Self]:
        return functools.partial(cls, *args, **kwargs)


class DomainIntersection(AbstractDomain):
    @property
    def domain(self) -> DomainIntersection:
        return self

    def __init__(self, datasets: Iterable[DependentDataset], scale: Mesoscale) -> None:
        (
            self._bbox,
            self._time_slice,
            self._time,
            self._levels,
            self._dvars,
            self._area_extents,
            self._datasets,
        ) = _get_intersection(datasets, scale)
        self._scale = scale

        self._datasets = DatasetSequence(
            ds.sel({LVL: [lvl]}) for lvl in self._levels for ds in self._datasets if lvl in ds.level
        )

    def __repr__(self) -> str:
        items = self._repr_args()
        return utils.join_kv("DomainIntersection:", *items)


# =====================================================================================================================
# - core functions for determining the domain of datasets
# =====================================================================================================================


def _fit_lon_lat(
    self: Iterable[DependentDataset],
    bbox: BoundingBox,
) -> Iterator[DependentDataset]:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (_mask_single_dataset_lon_lat(ds, min_lon, max_lon, min_lat, max_lat) for ds in self)


def _mask_single_dataset_lon_lat(
    ds: DependentDataset,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
) -> DependentDataset:
    lons = (ds.lons.to_numpy() + 180.0) % 360 - 180.0
    lats = ds.lats.to_numpy()
    x_mask = (lons >= min_lon) & (lons <= max_lon)
    y_mask = (lats >= min_lat) & (lats <= max_lat)
    ds = ds.sel({X: x_mask.any(axis=0), Y: y_mask.any(axis=1)})
    return ds.set_grid_definition()


def _batch_time(self: Iterable[DependentDataset], batcher: Array[[N, N], np.datetime64]) -> Iterator[DependentDataset]:
    return (ds.sel({TIME: ds[TIME].isin(t)}) for t in batcher for ds in self)


def _get_intersection(
    dsets: Iterable[DependentDataset], scale: Mesoscale
) -> tuple[
    BoundingBox,
    TimeSlice,
    Array[[N], np.datetime64],
    Array[[N], np.float_],
    Array[[N, N], np.str_],
    Array[[N, N4], np.float_],
    DatasetSequence,
]:
    dsets = DatasetSequence(dsets)
    coords = dsets.get_coordinates()

    # - Z
    levels = np.concatenate(coords[LVL])

    # - T
    time = utils.overlapping(coords[TIME], sort=True)
    dsets = DatasetSequence(ds.sel({TIME: ds.time.isin(time)}) for ds in dsets)
    min_time, max_time = time[0], time[-1]

    # - XY
    (min_lon, max_lon) = _min_max(coords[LON])
    (min_lat, max_lat) = _min_max(coords[LAT])

    return (
        BoundingBox(min_lon, min_lat, max_lon, max_lat),
        np.s_[min_time:max_time],
        time,
        np.sort(levels[np.isin(levels, scale.levels)])[::-1],  # descending,
        np.stack([list(ds.data_vars) for ds in dsets]),
        scale.area_extent * 1000.0,  # km -> m
        dsets,
    )


def _min_max(arr: list[Array[[...], np.float_]]) -> tuple[float, float]:
    mins = []
    maxs = []
    for x in arr:
        mins.append(x.min())
        maxs.append(x.max())
    mn, mx = np.max(mins), np.min(maxs)
    return (mn, mx)
