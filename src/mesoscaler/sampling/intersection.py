# @dataclasses.dataclass(frozen=True)
from __future__ import annotations

import abc
import itertools
from typing import Iterable

import numpy as np

from .._typing import (
    TYPE_CHECKING,
    Any,
    AreaExtent,
    Array,
    Iterable,
    Iterator,
    N,
    Self,
    TimeSlice,
)
from ..enums import LVL, TIME
from ..utils import slice_time

if TYPE_CHECKING:
    from ..core import DependentDataset, Mesoscale


DatasetAndExtent = tuple["DependentDataset", AreaExtent]


class AbstractIntersection(abc.ABC):
    @property
    @abc.abstractmethod
    def intersection(self) -> DatasetIntersection:
        ...

    @property
    def min_lon(self) -> float:
        return self.intersection._min_lon

    @property
    def max_lon(self) -> float:
        return self.intersection._max_lon

    @property
    def min_lat(self) -> float:
        return self.intersection._min_lat

    @property
    def max_lat(self) -> float:
        return self.intersection._max_lat

    @property
    def min_time(self) -> np.datetime64:
        return self.intersection._min_time

    @property
    def max_time(self) -> np.datetime64:
        return self.intersection._max_time

    @property
    def time(self) -> Array[[N], np.datetime64]:
        return self.intersection._time

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self.intersection._levels

    @property
    def dvars(self) -> Array[[N, N], np.str_]:
        return self.intersection._dvars

    @property
    def area_extents(self) -> AreaExtent:
        return self.intersection._area_extents

    @property
    def datasets(self) -> Iterable[DependentDataset]:
        return self.intersection._datasets

    def get_min_max(self, key: str) -> tuple[float, float]:
        if key == "lon":
            return self.min_lon, self.max_lon
        elif key == "lat":
            return self.min_lat, self.max_lat
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


class DatasetIntersection(AbstractIntersection):
    @property
    def intersection(self) -> DatasetIntersection:
        return self

    def __init__(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        min_time: np.datetime64,
        max_time: np.datetime64,
        *,
        time: Array[[N], np.datetime64],
        levels: Array[[N], np.float_],
        dvars: Array[[N, N], np.str_],
        area_extents: AreaExtent,
        datasets: Iterable[DependentDataset],
    ):
        self._min_lon = min_lon
        self._max_lon = max_lon
        self._min_lat = min_lat
        self._max_lat = max_lat
        self._min_time = min_time
        self._max_time = max_time
        self._time = time
        self._levels = levels
        self._dvars = dvars
        self._area_extents = area_extents
        self._datasets = tuple(
            ds.sel({TIME: np.s_[min_time:max_time], LVL: [lvl]})  # type: ignore
            for lvl in levels
            for ds in datasets
            if lvl in ds.level
        )

    @classmethod
    def from_datasets(cls, datasets: Iterable[DependentDataset], scale: Mesoscale) -> Self:
        datasets = tuple(datasets)
        lons = []
        lats = []
        time = []  # type: list[Array[[N], np.datetime64]]
        levels = []
        dvars = []
        for ds in datasets:
            lons.append((ds.lons.to_numpy() - 180) % 360 - 180)
            lats.append(ds.lats.to_numpy())
            time.append(ds.time.to_numpy())
            levels.append(ds.level.to_numpy())
            dvars.append(list(ds.data_vars))

        chain = itertools.chain.from_iterable(map(cls._min_max, (lons, lats, time)))
        levels = np.concatenate(levels)

        return cls(
            *chain,
            time=np.sort(np.unique(np.concatenate(time))),
            levels=np.sort(levels[np.isin(levels, scale.levels)])[::-1],  # descending,
            dvars=np.stack(dvars),
            area_extents=scale.area_extent * 1000.0,  # km -> m
            datasets=datasets,
        )

    @staticmethod
    def _min_max(arr: list[np.ndarray]) -> tuple[Any, Any]:
        mins = []
        maxs = []
        for x in arr:
            mins.append(x.min())
            maxs.append(x.max())
        mn, mx = np.max(mins), np.min(maxs)
        return (mn, mx)
