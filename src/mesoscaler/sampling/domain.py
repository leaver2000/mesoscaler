# @dataclasses.dataclass(frozen=True)
from __future__ import annotations

import abc
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
    Sequence,
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
from ..generic import DataSequence, Mapping

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

    def nd_intersect(
        self, lon: AnyArrayLike[np.float_], lat: AnyArrayLike[np.float_]
    ) -> dict[Dimensions, AnyArrayLike[np.bool_]]:
        x0, y0, x1, y1 = self
        x0 = (x0 - 180.0) % 360 + 180.0
        x1 = (x1 - 180.0) % 360 + 180.0
        return {X: ((lon >= x0) & (lon <= x1)).any(axis=0), Y: ((lat >= y0) & (lat <= y1)).any(axis=1)}


class DatasetSequence(DataSequence[DependentDataset]):
    def get_domain(self, scale: Mesoscale) -> Domain:
        return scale.get_domain(self)

    def get_time(self) -> list[Array[[N], np.datetime64]]:
        return [ds[TIME].to_numpy() for ds in self]

    def get_levels(self) -> list[Array[[N], np.float_]]:
        return [ds[LVL].to_numpy() for ds in self]

    def get_longitude(self) -> list[Array[[N, N], np.float_]]:
        return [((ds[LON].to_numpy() - 180.0) % 360 - 180.0) for ds in self]

    def get_latitude(self) -> list[Array[[N, N], np.float_]]:
        return [ds[LAT].to_numpy() for ds in self]

    def _repr_html_(self) -> str:
        return "\n".join(ds._repr_html_() for ds in self)

    # def sel(self, time=None, levels=None) -> DatasetSequence:
    #     keys = []  # type: list[tuple[str, Any]]
    #     if time is not None:
    #         keys.append((TIME, time))
    #     if levels is not None:
    #         keys.append((LVL, levels))
    #     return DatasetSequence(ds.sel({key: ds[key].isin(value)}) for ds in self for key, value in keys)


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
        return self.scale.to_numpy(units="m")

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
    def datasets(self) -> DatasetSequence:
        return self.domain._datasets

    def iter_dataset_and_extent(self) -> Iterator[DatasetAndExtent]:
        return zip(self.datasets, self.area_extents)

    def _repr_args(self):
        return (
            ("bbox", utils.repr_(self.bbox)),
            ("time", utils.repr_(self.time)),
            ("scale", "\n" + textwrap.indent(utils.repr_(self.scale), "  ")),
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

    def fit(self, __x: Iterable[DependentDataset]) -> DatasetSequence:
        return DatasetSequence(
            ds.sel({LVL: [lvl]}).set_grid_definition(grid)
            for lvl in self.levels
            for (ds, grid) in ((ds, ds.get_grid_definition()) for ds in __x)
            if lvl in ds.level
        )

    def batch(self, x: Mapping[Coordinates, Iterable[Sequence[float | np.datetime64]]]) -> DatasetSequence:
        return DatasetSequence(
            ds.select_from({k: batch}) for ds in self.datasets for k, batcher in x.items() for batch in batcher
        )


class Domain(AbstractDomain):
    @property
    def domain(self) -> Domain:
        return self

    def __init__(self, datasets: Iterable[DependentDataset], scale: Mesoscale) -> None:
        (
            self._bbox,
            self._time,
            self._levels,
            self._dvars,
            self._area_extents,
            self._datasets,
            self._scale,
        ) = _get_intersection(datasets, scale)
        super().__init__()

        self._datasets = self.fit(self._datasets)

    def __repr__(self) -> str:
        chain_items = self._repr_args()
        return utils.join_kv("Domain:", *chain_items)


# =====================================================================================================================
# - core functions for determining the domain of datasets
# =====================================================================================================================


def _get_intersection(
    dsets: Iterable[DependentDataset], scale: Mesoscale
) -> tuple[
    BoundingBox,
    Array[[N], np.datetime64],
    Array[[N], np.float_],
    Array[[N, N], np.str_],
    Array[[N, N4], np.float_],
    DatasetSequence,
    Mesoscale,
]:
    dsets = DatasetSequence(dsets)
    # coords = dsets.get_coordinates()

    # - Z
    levels = utils.nd_union(dsets.get_levels(), sort=True)
    levels = levels[np.isin(levels, scale.levels, assume_unique=True)][::-1]  # descending,

    # - T
    time = utils.nd_intersect(dsets.get_time(), sort=True)
    dsets = DatasetSequence(ds.sel({TIME: ds.time.isin(time)}) for ds in dsets)

    # - XY
    (x0, x1) = _min_max(dsets.get_longitude())
    (y0, y1) = _min_max(dsets.get_latitude())

    return (
        BoundingBox(x0, y0, x1, y1),
        # np.s_[min_time:max_time],
        time,
        levels,
        np.stack([list(ds.data_vars) for ds in dsets]),
        scale.area_extent * 1000.0,  # km -> m
        dsets,
        scale,
    )


def _min_max(arr: list[Array[[N, N], np.float_]]) -> tuple[float, float]:
    mins = []
    maxs = []
    for x in arr:
        mins.append(x.min())
        maxs.append(x.max())
    mn, mx = np.max(mins), np.min(maxs)
    return (mn, mx)
