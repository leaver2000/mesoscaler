from __future__ import annotations

import abc
import functools
import itertools

import numpy as np

from .. import _compat as compat, utils
from .._typing import (
    Any,
    Array,
    Callable,
    Iterator,
    N,
    Nx,
    Ny,
    Point,
    PointOverTime,
    Self,
    TypeVar,
)
from ..generic import DataSampler
from .domain import AbstractDomain, BoundingBox, Domain

_T = TypeVar("_T")


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


class TimeAndPointSampler(DomainIntersectionSampler[PointOverTime], abc.ABC):
    _indices: list[PointOverTime] | None

    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(self, domain: Domain, time_step: int) -> None:
        super().__init__(domain)
        self._indices = None
        self._lon_lats = None
        self._time_batches = None
        self.time_step = time_step

    @property
    def time_batches(self) -> Array[[N, N], np.datetime64]:
        if self._time_batches is None:
            self._time_batches = time_batches = self.get_time_batches()
            return time_batches
        return self._time_batches

    @property
    def lon_lats(self) -> tuple[Array[[Nx], np.float_], Array[[Ny], np.float_]]:
        if self._lon_lats is None:
            self._lon_lats = lon_lats = self.get_lon_lats()
            return lon_lats
        return self._lon_lats

    @property
    def indices(self) -> list[PointOverTime]:
        if self._indices is None:
            self._indices = indices = list(self.iter_indices())
            return indices
        return self._indices

    # =================================================================================================================
    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_samples(self) -> int:
        return len(self)

    def __repr__(self) -> str:
        indices = "\n".join(utils.repr_(self.indices, map_values=True))
        return f"{type(self).__name__}[\n{indices}\n]"

    # =================================================================================================================
    @abc.abstractmethod
    def get_lon_lats(self) -> tuple[Array[[Nx], np.float_], Array[[Ny], np.float_]]:
        ...

    def get_time_batches(self) -> Array[[N, N], np.datetime64]:
        return utils.batch(self.times, self.time_step, strict=True).astype("datetime64[h]")

    # =================================================================================================================
    def iter_time(self) -> Iterator[Array[[N], np.datetime64]]:
        yield from self.get_time_batches()

    def iter_points(self) -> Iterator[Point]:
        lons, lats = self.get_lon_lats()
        yield from itertools.product(lons, lats)

    def iter_indices(self) -> Iterator[PointOverTime]:
        yield from itertools.product(self.iter_points(), self.iter_time())

    def __iter__(self) -> Iterator[PointOverTime]:
        yield from compat.tqdm(self.iter_indices())


class MultiPointSampler(TimeAndPointSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(self, domain: Domain, /, *points: tuple[float, float], time_step: int) -> None:
        super().__init__(domain, time_step)
        self.time_step = time_step
        self._points = tuple(dict.fromkeys(points))

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        x, y = np.array(self._points).T
        return x, y

    def iter_points(self) -> Iterator[Point]:
        # dont create a product of points
        return iter(self._points)


class AreaOfInterestSampler(TimeAndPointSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    @staticmethod
    def _resolve_stride(
        stride: int | tuple[int, ...] | None, time_step: int | None, lon_lat_step: int | tuple[int, ...] | None
    ) -> tuple[int, int, int]:
        if isinstance(stride, tuple) and len(stride) == 3:
            if lon_lat_step is not None or time_step is not None:
                import warnings

                warnings.warn(
                    f"stride {stride} is overriding time_step {time_step} and lon_lat_step {lon_lat_step}",
                    RuntimeWarning,
                )
            stride = stride
        elif stride is None:
            if lon_lat_step is None:
                stride = (time_step or 1, 1, 1)
            elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
                stride = (time_step or 1, *lon_lat_step)
            elif isinstance(lon_lat_step, int):
                stride = (time_step or 1, lon_lat_step, lon_lat_step)
            else:
                raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")
        elif isinstance(stride, int):
            if lon_lat_step is None:
                stride = (time_step or 1, stride, stride)
            elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
                stride = (time_step or 1, *lon_lat_step)
            elif isinstance(lon_lat_step, int):
                stride = (time_step or 1, lon_lat_step, lon_lat_step)
            else:
                raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")

        elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
            stride = (time_step or 1, *lon_lat_step)
        else:
            raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")

        return stride

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        stride: int | tuple[int, ...] | None = None,
        aoi: tuple[float, float, float, float] | None = None,
        padding: tuple[float, float] | float | None = 2.5,
        # - convince args
        time_step: int | None = None,
        lon_lat_step: int | tuple[int, ...] | None = None,
    ) -> None:
        stride = self._resolve_stride(stride, time_step, lon_lat_step)
        super().__init__(domain, stride[0])
        self.stride = stride
        self.aoi = BoundingBox(*(self.bbox if aoi is None else aoi))
        self.padding = (padding, padding) if padding is not None and not isinstance(padding, tuple) else padding

    def get_time_batches(self) -> Array[[N, N], np.datetime64]:
        t_num = self.stride[0]
        return utils.batch(self.times, t_num, strict=True).astype("datetime64[h]")

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        _, x_num, y_num = self.stride
        x_min, y_min, x_max, y_max = (
            self.aoi.pad(self.padding) if self.padding is not None else self.aoi
        )  # self._get_padded_aoi(self.padding) if self.padding else self.aoi
        if self.domain.min_lat > y_min or self.domain.max_lat < y_max:
            raise ValueError(f"area_extent latitude bounds {y_min, y_max} are outside dataset bounds")
        elif self.domain.min_lon > x_min or self.domain.max_lon < x_max:
            raise ValueError(f"area_extent longitude bounds {x_min, x_max} are outside dataset bounds")

        return np.linspace(x_min, x_max, x_num), np.linspace(y_min, y_max, y_num)


DEFAULT_SAMPLER = functools.partial(
    AreaOfInterestSampler,
    stride=(2, 5, 5),
    aoi=(-125.0, 45.0, -65.0, 25.0),
)
