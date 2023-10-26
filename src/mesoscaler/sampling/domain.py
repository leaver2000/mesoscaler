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
)
from ..enums import LAT, LON, LVL, TIME, Dimensions, X, Y
from ..generic import DataSequence

if TYPE_CHECKING:
    from ..core import DependentDataset, Mesoscale
else:
    Mesoscale = Any
    DependentDataset = Any


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


class AbstractDomain(abc.ABC):
    @property
    @abc.abstractmethod
    def domain(self) -> Domain:
        ...

    @property
    def scale(self) -> Mesoscale:
        return self.domain._scale

    @property  # - T
    def time(self) -> Array[[N], np.datetime64]:
        return self.domain._time

    @property  # - Z
    def levels(self) -> Array[[N], np.float_]:
        return self.domain._levels

    @property  # - YX
    def bbox(self) -> BoundingBox:
        return self.domain._bbox

    @property
    def datasets(self) -> DatasetSequence:
        return self.domain._datasets

    # - Extent
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

    @property
    def area_extent(self) -> Array[[N, N4], np.float_]:
        return self.scale.to_numpy(units="m")

    def iter_dataset_and_extent(self) -> Iterator[tuple[DependentDataset, AreaExtent]]:
        return zip(self.datasets, self.area_extent)

    def _repr_args(self):
        return (
            ("bbox", utils.repr_(self.bbox)),
            ("time", utils.repr_(self.time)),
            ("scale", "\n" + textwrap.indent(utils.repr_(self.scale), "  ")),
        )


class Domain(AbstractDomain):
    __slots__ = ("_scale", "_time", "_levels", "_bbox", "_datasets")
    _scale: Mesoscale
    _time: Array[[N], np.datetime64]
    _levels: Array[[N], np.float_]
    _bbox: BoundingBox
    _datasets: DatasetSequence

    @property
    def domain(self) -> Domain:
        return self

    def __init__(self, dsets: Iterable[DependentDataset], scale: Mesoscale) -> None:
        self._scale = scale
        dsets = DatasetSequence(dsets) if not isinstance(dsets, DatasetSequence) else dsets

        # - Z
        levels = utils.nd_union(dsets.get_levels(), sort=True)
        self._levels = levels = levels[np.isin(levels, scale.levels, assume_unique=True)][::-1]  # descending,

        # - T
        self._time = time = utils.nd_intersect(dsets.get_time(), sort=True)

        # - XY
        (x0, x1) = _min_max(dsets.get_longitude())
        (y0, y1) = _min_max(dsets.get_latitude())
        self._bbox = BoundingBox(x0, y0, x1, y1)
        self._datasets = DatasetSequence(self._fit(dsets, time, levels))

        super().__init__()

    def __repr__(self) -> str:
        items = self._repr_args()
        return utils.join_kv("Domain:", *items)

    @staticmethod
    def _fit(
        dsets: Iterable[DependentDataset], time: Array[[N], np.datetime64], levels: Array[[N], np.float_]
    ) -> Iterable[DependentDataset]:
        dsets = (ds.sel({TIME: ds.time.isin(time)}) for ds in dsets)
        for ds in dsets:
            grid = ds.get_grid_definition()
            for lvl in levels:
                if lvl in ds.level:
                    yield ds.sel({LVL: [lvl]}).set_grid_definition(grid)

    def fit(self, dsets: Iterable[DependentDataset], /) -> DatasetSequence:
        return DatasetSequence(self._fit(dsets, self.time, self.levels))


# =====================================================================================================================
# - core functions for determining the domain of datasets
# =====================================================================================================================


def _min_max(arr: list[Array[[N, N], np.float_]]) -> tuple[float, float]:
    mins = []
    maxs = []
    for x in arr:
        mins.append(x.min())
        maxs.append(x.max())
    mn, mx = np.max(mins), np.min(maxs)
    return (mn, mx)
