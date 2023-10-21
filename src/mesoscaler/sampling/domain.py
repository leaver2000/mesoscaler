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
    AnyArrayLike,
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
from ..enums import (
    LAT,
    LON,
    LVL,
    TIME,
    Coordinates,
    Dimensions,
    DimensionsMapType,
    X,
    Y,
    Z,
)
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
from pandas._typing import AnyArrayLike


class BoundingBox(NamedTuple):
    """
    >>> min_lon, min_lat, max_lon, max_lat = -180.0, 180.0, -90.0, 90.0
    >>> x0, y0, x1, y1 = west, south, east, north = BoundingBox(min_lon, min_lat, max_lon, max_lat)
    >>>
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

    @property
    def x(self) -> tuple[float, float]:
        return self.west, self.east

    @property
    def y(self) -> tuple[float, float]:
        return self.south, self.north

    @property
    def xy0(self) -> tuple[float, float]:
        return self.west, self.south

    @property
    def xy1(self) -> tuple[float, float]:
        return self.east, self.north

    def linspace(self, key: str, *, frequency: int, round: int = 5) -> Array[[N], Any]:
        if key.upper() == X:
            start, stop = self.x
        elif key.upper() == Y:
            start, stop = self.y
        else:
            raise KeyError(f"key {key} is not supported!")
        return np.linspace(start, stop, frequency).round(round)

    def meshgrid(self, frequency: int) -> tuple[Array[[N, N], np.float_], Array[[N, N], np.float_]]:
        x, y = np.meshgrid(self.linspace(X, frequency=frequency), self.linspace(Y, frequency=frequency))
        return x, y

    def __array__(self) -> Array[[N4], np.float_]:
        return np.array(self)

    def to_numpy(self) -> Array[[N4], np.float_]:
        return np.array(self)

    def intersect2d(
        self, lon: AnyArrayLike[np.float_], lat: AnyArrayLike[np.float_]
    ) -> dict[Dimensions, AnyArrayLike[np.bool_]]:
        x0, y0, x1, y1 = self
        x0 = (x0 - 180.0) % 360 + 180.0
        x1 = (x1 - 180.0) % 360 + 180.0
        return {X: ((lon >= x0) & (lon <= x1)).any(axis=0), Y: ((lat >= y0) & (lat <= y1)).any(axis=1)}


class DatasetSequence(DataSequence[DependentDataset]):
    def get_coordinates(self) -> Mapping[Coordinates, Sequence[Array[[...], Any]]]:
        return {
            LON: [((ds[LON].to_numpy() - 180.0) % 360 - 180.0) for ds in self],
            LAT: [ds[LAT].to_numpy() for ds in self],
            TIME: [ds[TIME].to_numpy() for ds in self],
            LVL: [ds[LVL].to_numpy() for ds in self],
        }

    def get_domain(self, scale: Mesoscale) -> Domain:
        return scale.get_domain(self)

    def batch_time(self, batcher: Array[[N, N], np.datetime64]) -> DatasetSequence:
        return DatasetSequence(_batch_time(self, batcher))

    def fit(self, x: AbstractDomain | Mesoscale, /) -> DatasetSequence:
        domain = x if isinstance(x, AbstractDomain) else self.get_domain(x)

        return DatasetSequence(
            ds.sel(
                {TIME: ds[TIME].isin(domain.time), LVL: ds[LVL].isin(domain.levels)}
                | domain.bbox.intersect2d(ds[LON], ds[LAT])
            ).set_grid_definition()
            for ds in self
        )

    def batch(self, x: Mapping[Coordinates, Iterable[Sequence[float | np.datetime64]]]) -> DatasetSequence:
        return DatasetSequence(
            ds.select_from({k: batch}) for ds in self for k, batcher in x.items() for batch in batcher
        )

    def _repr_html_(self) -> str:
        return "\n".join(ds._repr_html_() for ds in self)


class AbstractDomain(abc.ABC):
    @property
    @abc.abstractmethod
    def domain(self) -> Domain:
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

    x0, y0, x1, y1 = min_lon, min_lat, max_lon, max_lat

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


class DomainIntersectionSampler(DataSampler[_T], AbstractDomain, abc.ABC):
    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(self, domain: Domain) -> None:
        super().__init__()
        self._domain = domain

    @classmethod
    def partial(cls, *args: Any, **kwargs: Any) -> Callable[[Domain], Self]:
        return functools.partial(cls, *args, **kwargs)


class Domain(AbstractDomain):
    @property
    def domain(self) -> Domain:
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
        return utils.join_kv("Domain:", *items)


# =====================================================================================================================
# - core functions for determining the domain of datasets
# =====================================================================================================================
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
    time = utils.overlapping(coords[TIME], sort=True)  # type: Array[[N], np.datetime64]
    dsets = DatasetSequence(ds.sel({TIME: ds.time.isin(time)}) for ds in dsets)
    min_time, max_time = time[0], time[-1]

    # - XY
    (x0, x1) = _min_max(coords[LON])
    (y0, y1) = _min_max(coords[LAT])

    return (
        BoundingBox(x0, y0, x1, y1),
        np.s_[min_time:max_time],
        time,
        np.sort(levels[np.isin(levels, scale.levels)])[::-1],  # descending,
        np.stack([list(ds.data_vars) for ds in dsets]),
        scale.area_extent * 1000.0,  # km -> m
        dsets,
    )


def _min_max(arr: Sequence[Array[[...], np.float_]]) -> tuple[float, float]:
    mins = []
    maxs = []
    for x in arr:
        mins.append(x.min())
        maxs.append(x.max())
    mn, mx = np.max(mins), np.min(maxs)
    return (mn, mx)
