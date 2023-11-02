from __future__ import annotations

import abc
import functools
import itertools
import textwrap

import numpy as np

from .. import _compat as compat, utils
from .._typing import (
    Any,
    Array,
    Callable,
    Iterable,
    Iterator,
    N,
    Nx,
    Ny,
    Point,
    PointOverTime,
    Self,
)
from ..generic import DataSampler
from .domain import AbstractDomain, BoundingBox, Domain


class PointOverTimeSampler(DataSampler[PointOverTime], AbstractDomain, abc.ABC):
    _indices: list[PointOverTime] | None

    @classmethod
    def partial(cls, *args: Any, **kwargs: Any) -> Callable[[Domain], Self]:
        return functools.partial(cls, *args, **kwargs)

    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(self, domain: Domain, time_step: int) -> None:
        super().__init__()
        self._domain = domain
        self.time_step = time_step
        self._indices = None
        self._lon_lats = None
        self._time_batches = None

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
        num_samples = self.num_samples
        if num_samples > 10:
            indices = "\n".join(utils.repr_(self.indices[:5], map_values=True))
            indices += "\n...\n"
            indices += "\n".join(utils.repr_(self.indices[-5:], map_values=True))
        else:
            indices = "\n".join(utils.repr_(self.indices, map_values=True))

        indices = textwrap.indent(indices, "    ")
        return f"{type(self).__name__}({num_samples=})[\n{indices}\n]"

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
        yield from self.iter_indices()

    def show(
        self,
        figsize: tuple[float, float] = (10, 10),
        *,
        fig: compat.plt.Figure | None = None,
        projection: compat.ccrs.Projection | compat.LiteralProjection | None = None,
        transform: compat.ccrs.Projection | None = None,
        marker: str = "o",
        color: str = "red",
        linewidth: float = 1.0,
        features: list[compat.cfeature.Feature] | None = None,
        **kwargs: Any,
    ) -> Self:
        if not compat.has_cartopy:
            raise RuntimeError("cartopy not installed!")

        extents = lon_0, lat_0, lon_1, lat_1 = self._get_extent()

        figsize = figsize or (10, 10)

        fig = fig or compat.plt.figure(figsize=figsize)
        if isinstance(projection, str):
            projection = compat.get_projection(
                projection, central_longitude=(lon_0 + lon_1) / 2, central_latitude=(lat_0 + lat_1) / 2
            )

            transform = transform or compat.ccrs.PlateCarree()
        elif projection is None:
            projection = compat.ccrs.PlateCarree()

        transform = transform or projection
        features = features or [compat.cfeature.STATES]
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        assert isinstance(ax, compat.GeoAxes)
        ax.coastlines()
        ax.set_extent(extents, crs=transform)

        for feature in features:
            ax.add_feature(feature)

        ax.scatter(
            *zip(*self.iter_points()),
            transform=transform,
            marker=marker,
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
        return self

    def _get_extent(self) -> list[float]:
        return [self.min_lon, self.max_lon, self.min_lat, self.max_lat]

    def chain(self, samplers: Iterable[PointOverTimeSampler]) -> ChainSampler:
        return ChainSampler((self, *samplers))


class ChainSampler(PointOverTimeSampler):
    def __init__(self, samplers: Iterable[PointOverTimeSampler]) -> None:
        domain = None
        time_step = None
        samplers, it = itertools.tee(samplers, 2)

        for s in it:
            if domain is None:
                domain = s.domain
            elif domain is not s.domain:
                raise ValueError(f"all samplers must have the same domain {domain=}")

            if time_step is None:
                time_step = s.time_step
            elif time_step != s.time_step:
                raise ValueError(f"all samplers must have the same time_step {time_step=}")

        assert domain is not None
        assert time_step is not None

        super().__init__(domain, time_step)
        self._points = itertools.chain.from_iterable(s.iter_points() for s in samplers)

    def iter_points(self) -> Iterator[Point]:
        # dont consume the chain
        it, self._points = itertools.tee(self._points, 2)
        yield from it

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        x, y = np.array(list(self.iter_points())).T
        return x, y


class MultiPointSampler(PointOverTimeSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(self, domain: Domain, /, *points: tuple[float, float] | Point, time_step: int) -> None:
        super().__init__(domain, time_step)
        self.time_step = time_step
        self._points = utils.point_union(points)

    def iter_points(self) -> Iterator[Point]:
        # dont create a product of points
        yield from self._points

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        x, y = np.array(list(self.iter_points())).T
        return x, y


class AreaOfInterestSampler(PointOverTimeSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    @staticmethod
    def _resolve_stride(
        stride: int | tuple[int, ...] | None, lon_lat_step: int | tuple[int, ...] | None, time_step: int | None
    ) -> tuple[int, int, int]:
        if isinstance(stride, tuple) and len(stride) == 3:
            if lon_lat_step is not None or time_step is not None:
                import warnings

                warnings.warn(
                    f"stride {stride} is overriding time_step {time_step} and lon_lat_step {lon_lat_step}",
                    RuntimeWarning,
                )
            stride = stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            xy, t = stride
            stride = (xy, xy, t)
        elif stride is None:
            if lon_lat_step is None:
                stride = (1, 1, time_step or 1)
            elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
                stride = (*lon_lat_step, time_step or 1)
            elif isinstance(lon_lat_step, int):
                stride = (lon_lat_step, lon_lat_step, time_step or 1)
            else:
                raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")
        elif isinstance(stride, int):
            if lon_lat_step is None:
                stride = (stride, stride, time_step or 1)
            elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
                stride = (*lon_lat_step, time_step or 1)
            elif isinstance(lon_lat_step, int):
                stride = (lon_lat_step, lon_lat_step, time_step or 1)
            else:
                raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")

        elif isinstance(lon_lat_step, tuple) and len(lon_lat_step) == 2:
            stride = (*lon_lat_step, time_step or 1)
        else:
            raise ValueError(f"stride {stride} and lon_lat_step {lon_lat_step} are not valid")

        return stride

    def _get_extent(self):
        x0, y0, x1, y1 = self.aoi
        return [x0, x1, y0, y1]

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        stride: int | tuple[int, ...] | None = None,
        aoi: tuple[float, float, float, float] | None = None,
        padding: tuple[float, float] | float | None = None,
        # - convince args
        lon_lat_step: int | tuple[int, ...] | None = None,
        time_step: int | None = None,
    ) -> None:
        stride = self._resolve_stride(stride, lon_lat_step, time_step)
        super().__init__(domain, stride[-1])
        self.stride = stride
        self.aoi = BoundingBox(*(self.bbox if aoi is None else aoi))
        self.padding = (padding, padding) if padding is not None and not isinstance(padding, tuple) else padding

    def get_time_batches(self) -> Array[[N, N], np.datetime64]:
        t_num = self.stride[-1]
        return utils.batch(self.times, t_num, strict=True).astype("datetime64[h]")

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        x_num, y_num = self.stride[:2]
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
