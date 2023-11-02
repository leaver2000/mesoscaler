from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Callable, Generic, TypeVar

import numpy as np
import pyresample.geometry
import xarray as xr
import zarr
from pyresample.geometry import AreaDefinition, GridDefinition

from .. import _compat as compat, utils
from .._typing import (
    Any,
    Array,
    Callable,
    DictStrAny,
    Final,
    Iterable,
    Iterator,
    Latitude,
    Literal,
    Longitude,
    Mapping,
    N,
    Nt,
    Nv,
    Nx,
    Ny,
    Nz,
)
from ..enums import LVL, TIME, T, X, Y, Z
from .display import PlotArray
from .domain import AbstractDomain, Domain
from .sampler import AreaOfInterestSampler, MultiPointSampler, PointOverTimeSampler

_VARIABLES = "variables"
SamplerT = TypeVar("SamplerT", bound=PointOverTimeSampler, contravariant=True)


class AbstractResampler(AbstractDomain, abc.ABC):
    _axes_transposed: Final[tuple[int, ...]] = (3, 4, 0, 1, 2)
    """
    >>> assert tuple(map(((Z, Y, X, C, T)).index, (C, T, Z, Y, X))) == ReSampler._axes_transposed
    """

    @property
    @abc.abstractmethod
    def resampler(self) -> ReSampler:
        """The root resampler."""

    @property
    def domain(self) -> Domain:
        return self.resampler._domain

    @property
    def height(self) -> int:
        return self.resampler._height

    @property
    def width(self) -> int:
        return self.resampler._width

    @property
    def proj(self) -> Literal["laea", "lcc"]:
        return self.resampler._proj

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        """
        Resamples the data cube at the given longitude, latitude, and time coordinates.

        Args:
            longitude: The longitude coordinates of the data cube.
            latitude: The latitude coordinates of the data cube.
            time: The time coordinates of the data cube.

        Returns:
            The resampled data cube, with dimensions (C, T, Z, Y, X), where:
            - C is the number of channels in the original data cube.
            - T is the number of time steps in the original data cube.
            - Z is the number of vertical levels in the resampled data cube.
            - Y is the number of rows in the resampled data cube.
            - X is the number of columns in the resampled data cube.
        """
        arr = self.resampler.zstack(longitude, latitude, time)  # (Z, Y, X, C*T)
        z, y, x = arr.shape[:3]
        # unsqueeze C
        arr = arr.reshape((z, y, x, -1, len(time)))  # (Z, Y, X, C, T)
        return arr.transpose(self._axes_transposed)  # (C, T, Z, Y, X)

    def get_array(self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]) -> xr.DataArray:
        data = self(longitude, latitude, time)

        return xr.DataArray(
            data,
            dims=(_VARIABLES, T, Z, Y, X),
            coords={
                # TODO: in order to set x and y we need to scale the extent
                # _VARIABLES: self.dvars[0],
                LVL: (LVL.axis, self.levels),
                TIME: (TIME.axis, time),
            },
        )

    def get_dataset(self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]) -> xr.Dataset:
        return self.get_array(longitude, latitude, time).to_dataset(_VARIABLES)


class ReSampler(AbstractResampler):
    # There are alot of callbacks and partial methods in this class.
    _proj: Literal["laea", "lcc"]

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
        sigmas=[1.0],
        radius_of_influence: int = 500_000,
        fill_value: int = 0,
        target_protection: Literal["laea", "lcc"] = "laea",
        reduce_data: bool = True,
        nprocs: int = 1,
        segments: Any = None,
        with_uncert: bool = False,
    ) -> None:
        super().__init__()
        self._domain = domain
        self._height = height
        self._width = width
        self._proj = target_protection
        self._resample_method = _get_resample_method(
            method,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
            reduce_data=reduce_data,
            nprocs=nprocs,
            segments=segments,
            with_uncert=with_uncert,
            sigmas=sigmas,
        )

    @property
    def resampler(self) -> ReSampler:
        return self

    @property
    def plot(self) -> functools.partial[PlotResampler]:
        return functools.partial(PlotResampler, self)

    @property
    def pipeline(self) -> Callable[[AreaOfInterestSampler], ResamplingPipeline]:
        return functools.partial(ResamplingPipeline, self)

    def create_pipeline(
        self,
        sampler: SamplerT | None = None,
        *points: tuple[float, float],
        aoi: tuple[float, float, float, float] | None = None,
        lon_lat_step: int = 5,
        time_step: int = 1,
        padding: tuple[float, float] | float | None = 2.5,
    ) -> ResamplingPipeline[SamplerT]:
        if sampler is not None:
            return ResamplingPipeline(self, sampler)
        return ResamplingPipeline.create(
            self, *points, aoi=aoi, lon_lat_step=lon_lat_step, time_step=time_step, padding=padding
        )

    def iter_area_definitions(self, longitude: Longitude, latitude: Latitude) -> Iterator[AreaDefinition]:
        pdef = self._partial_area_definition(longitude, latitude)
        for area_extent in self.area_extent:
            yield pdef(area_extent=area_extent)

    def get_area_definitions(self, longitude: Longitude, latitude: Latitude) -> list[AreaDefinition]:
        return list(self.iter_area_definitions(longitude, latitude))

    def get_area_definition_dict(self, longitude: Longitude, latitude: Latitude) -> dict[float, AreaDefinition]:
        return {lvl: ad for lvl, ad in zip(self.levels, self.iter_area_definitions(longitude, latitude))}

    # zstack -> _resample_point_over_time -> _partial_area_definition -> _area_definition

    def zstack(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nz, Ny, Nx, Nv | Nt], np.float_]:
        """
        Resamples the data at the given longitude, latitude, and time coordinates and returns a 4D array
        with the resampled data stacked along the first dimension.

        Args:
            longitude: The longitude coordinate of the point to resample.
            latitude: The latitude coordinate of the point to resample.
            time: An array of datetime64 objects representing the times at which to resample the data.

        Returns:
            A 4D array with the resampled data stacked along the first dimension. The shape of the array
            is (Z, Y, X, C*T), where Z is the number of resampled points, Y and X are the spatial dimensions
            of the resampled data, C is the number of channels in the data, and T is the number of time points.
        """
        return np.stack(self._resample_point_over_time(longitude, latitude, time))  # (Z, Y, X, C*T)

    def _resample_point_over_time(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> list[Array[[Ny, Nx, N], np.float_]]:
        """resample the data along the vertical scale for a single point over time.
        each item in the list is a 3-d array that can be stacked into along the vertical axis.

        The variables and Time are stacked into `N`.
        """
        pdef = self._partial_area_definition(longitude, latitude)

        return [
            self._resample_method(
                ds.grid_definition,
                # TODO: it would probably be beneficial to slice the data before resampling
                # to prevent loading all of th lat_lon data into memory
                ds.sel({TIME: time}).to_stacked_array("C", [Y, X]).to_numpy(),
                pdef(area_extent=area_extent),
            )
            for ds, area_extent in self.iter_dataset_and_extent()
        ]

    def _partial_area_definition(self, longitude: Longitude, latitude: Latitude) -> functools.partial[AreaDefinition]:
        return functools.partial(
            utils.area_definition,
            width=self._width,
            height=self._height,
            projection={"proj": self._proj, "lon_0": longitude, "lat_0": latitude},
        )


@dataclasses.dataclass
class ZarrAttributes:
    proj: Literal["laea", "lcc"]
    height: int
    width: int
    time_period: list[str]
    scaling: list[dict[str, Any]]
    metadata: list[dict[str, Any]]

    @classmethod
    def from_array(cls, array: zarr.Array | zarr.Group) -> ZarrAttributes:
        return ZarrAttributes(
            proj=array.attrs["proj"],
            height=array.attrs["height"],
            width=array.attrs["width"],
            time_period=array.attrs["time_period"],
            scaling=array.attrs["scaling"],
            metadata=array.attrs["metadata"],
        )


class ResamplingPipeline(Iterable[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]], AbstractDomain, Generic[SamplerT]):
    @property
    def domain(self) -> Domain:
        return self._resampler.domain

    def __init__(self, resampler: ReSampler, sampler: SamplerT) -> None:
        super().__init__()
        self._resampler = resampler
        self._sampler = sampler

    @classmethod
    def create(
        cls,
        resampler: ReSampler,
        *points: tuple[float, float],
        aoi: tuple[float, float, float, float] | None = None,
        lon_lat_step: int = 5,
        time_step: int = 1,
        padding: tuple[float, float] | float | None = 2.5,
    ) -> ResamplingPipeline[Any]:
        if points:
            sampler = MultiPointSampler(resampler.domain, *points, time_step=time_step)
        else:
            sampler = AreaOfInterestSampler(
                resampler.domain, aoi=aoi, lon_lat_step=lon_lat_step, time_step=time_step, padding=padding
            )
        return ResamplingPipeline(resampler, sampler)

    def __iter__(self) -> Iterator[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]:
        for (lon, lat), time in self._sampler:
            yield self._resampler(lon, lat, time)

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        s = self._sampler
        r = self._resampler

        return (
            s.num_samples,  # - B
            self.num_vars // len(self.datasets),  # - C
            s.time_step,  # - T
            s.num_levels,  # - Z
            r.height,  # - Y
            r.width,  # - X
        )

    @property
    def attributes(self) -> DictStrAny:
        data = {
            "shape": self.shape,
            "proj": self._resampler.proj,
            "height": self._resampler.height,
            "width": self._resampler.width,
        }
        data["time_period"] = (
            np.array([self.domain.times.min(), self.domain.times.max()]).astype("datetime64[h]").astype(str).tolist()
        )

        data["scaling"] = [
            {"scale": scl, "level": lvl, "extent": ext}
            for lvl, ext, scl in zip(
                self.domain.scale.levels.tolist(),
                self.domain.scale.to_numpy().tolist(),
                self.domain.scale.scale.tolist(),
            )
        ]

        return data

    def write_zarr(self, path: str, name: str | Mapping[str, int] = "data", chunk_size: int = 10) -> None:
        if not isinstance(name, str):
            # TODO: add means to shuffle and write multiple groups
            raise NotImplementedError("writing multiple groups is not supported yet!")

        resampler = self._resampler
        sampler = self._sampler
        shape = self.shape
        # - Create a zarr store
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store)

        # - Chunk size and shape
        # In general, chunks of at least 1 megabyte (1M)
        # uncompressed size seem to provide better performance,
        # at least when using the Blosc compression library.

        # The optimal chunk shape will depend on how you want to access the data.
        # E.g., for a 2-dimensional array, if you only ever take slices along the
        # first dimension, then chunk across the second dimension.
        # If you know you want to chunk across an entire dimension you can
        # use None or -1 within the chunks argument, e.g.:
        # z1 = zarr.zeros((10000, 10000), chunks=(100, None), dtype='i4')
        # z1.chunks
        # (100, 10000)
        g = root.create_dataset(
            name,
            shape=shape,
            # the arrays will be loaded as the 6 dimensional array
            # (sample, channel, time, level, height, width)
            chunks=(chunk_size, None, None, None, None, None),
            dtype=np.float32,
        )
        for k, v in self.attributes.items():
            g.attrs[k] = v
        metadata = []
        for i, ((lon, lat), time) in enumerate(compat.tqdm(sampler)):
            g[i, ...] = resampler(lon, lat, time)

            metadata.append(
                {
                    "longitude": lon,
                    "latitude": lat,
                    "time": time.astype("datetime64[h]").astype(str).tolist(),
                }
            )
            g.attrs["metadata"] = metadata


class PlotResampler(PlotArray, AbstractResampler):
    @property
    def resampler(self) -> ReSampler:
        return self._resampler

    @property
    def domain(self) -> Domain:
        return self.resampler.domain

    def __init__(
        self,
        resampler: ReSampler,
        longitude: Longitude,
        latitude: Latitude,
        time: Array[[N], np.datetime64],
        /,
        *,
        transform: compat.ccrs.Projection | None = None,
        features: list = [],
        grid_lines: bool = True,
        coast_lines: bool = True,
    ) -> None:
        self._resampler = resampler

        super().__init__(
            self.resampler(longitude, latitude, time),
            longitude,
            latitude,
            time,
            self.resampler.levels,
            self.resampler.width,
            self.resampler.height,
            self.resampler.get_area_definitions(longitude, latitude),
            transform=transform,
            features=features,
            grid_lines=grid_lines,
            coast_lines=coast_lines,
        )

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64], **kwargs
    ) -> PlotResampler:
        """Create a new BatchPlotter with the same resampler and batch data, but with a different center."""
        return self.resampler.plot(longitude, latitude, time, **(self._config | kwargs))


def _get_resample_method(
    method: str,
    radius_of_influence: int = 100_000,
    fill_value: int = 0,
    reduce_data: bool = True,
    nprocs: int = 1,
    segments=None,
    # - gauss -
    sigmas=[1.0],
    with_uncert: bool = False,
    neighbors: int = 8,
    epsilon: int = 0,
) -> Callable[[GridDefinition, Array[[Ny, Nx, N], np.float_], AreaDefinition], Array[[Ny, Nx, N], np.float_]]:
    kwargs = dict(
        radius_of_influence=radius_of_influence,
        fill_value=fill_value,
        reduce_data=reduce_data,
        nprocs=nprocs,
        segments=segments,
    )
    if method == "nearest":
        func = pyresample.kd_tree.resample_nearest

    elif method == "gauss":
        func = pyresample.kd_tree.resample_gauss
        kwargs |= dict(sigmas=sigmas, with_uncert=with_uncert, neighbours=neighbors, epsilon=epsilon)
    else:
        raise ValueError(f"method {method} is not supported!")
    return functools.partial(func, **kwargs)
