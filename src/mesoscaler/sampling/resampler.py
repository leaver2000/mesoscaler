from __future__ import annotations

import abc
import functools

import numpy as np
import pyproj
import pyresample.geometry
import xarray as xr
from pyresample.geometry import AreaDefinition, GridDefinition

from .._typing import (
    N4,
    Any,
    Array,
    Callable,
    Final,
    Latitude,
    Literal,
    Longitude,
    N,
    Nt,
    Nv,
    Nx,
    Ny,
    Nz,
    Sequence,
)
from ..enums import LVL, TIME, T, X, Y, Z
from .domain import AbstractDomain, Domain

_VARIABLES = "variables"


class AbstractResampler(AbstractDomain, abc.ABC):
    _axes_transposed: Final[tuple[int, ...]] = (3, 4, 0, 1, 2)
    """
    >>> assert tuple(map(((Z, Y, X, C, T)).index, (C, T, Z, Y, X))) == ReSampler._axes_transposed
    """

    @abc.abstractmethod
    def vstack(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nz, Ny, Nx, Nv | Nt], np.float_]:
        """Call the root resampler to resample the data for a single point over time.

        ```
        class SomeClass(AbstractResampler):
            resampler:Resampler = ...

            def vstack(
                self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
            ) -> Array[[Nz, Ny, Nx, Nv | Nt], np.float_]:
                return self.resampler.vstack(longitude, latitude, time)
        ```
        """

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        """stack the data along `Nz` -> unsqueeze the variables & time -> reshape the data to match the expected output."""

        arr = self.vstack(longitude, latitude, time)  # (Z, Y, X, C*T)
        z, y, x = arr.shape[:3]
        # unsqueeze C
        arr = arr.reshape((z, y, x, -1, time.size))  # (Z, Y, X, C, T)
        return arr.transpose(self._axes_transposed)  # (C, T, Z, Y, X)

    def get_array(self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]) -> xr.DataArray:
        data = self(longitude, latitude, time)

        return xr.DataArray(
            data,
            dims=(_VARIABLES, T, Z, Y, X),
            coords={
                _VARIABLES: self.dvars[0],
                LVL: (LVL.axis, self.levels),
                # TIME: (TIME.axis, self.slice_time(time)),
            },
        )

    def get_dataset(self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]) -> xr.Dataset:
        return self.get_array(longitude, latitude, time).to_dataset(_VARIABLES)


class ReSampler(AbstractResampler):
    # There are alot of callbacks and partial methods in this class.

    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        height: int = 80,
        width: int = 80,
        method: str = "nearest",
        sigmas=[1.0],
        radius_of_influence: int = 500000,
        fill_value: int = 0,
        target_protection: Literal["laea", "lcc"] = "laea",
        reduce_data: bool = True,
        nprocs: int = 1,
        segments: Any = None,
        with_uncert: bool = False,
    ) -> None:
        super().__init__()
        self._domain = domain
        self.height = height
        self.width = width
        self.proj = target_protection

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

    def _partial_area_definition(self, longitude: Longitude, latitude: Latitude) -> functools.partial[AreaDefinition]:
        return functools.partial(
            _area_definition,
            width=self.width,
            height=self.height,
            projection={"proj": self.proj, "lon_0": longitude, "lat_0": latitude},
        )

    def _resample_point_over_time(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> list[Array[[Ny, Nx, N], np.float_]]:
        """resample the data along the vertical scale for a single point over time.
        each item in the list is a 3-d array that can be stacked into along the vertical axis.

        The variables and Time are stacked into `N`.
        """
        area_definition = self._partial_area_definition(longitude, latitude)

        return [
            self._resample_method(
                ds.grid_definition,
                # TODO: it would probably be beneficial to slice the data before resampling
                # to prevent loading all of th lat_lon data into memory
                ds.sel({TIME: time}).to_stacked_array("C", [Y, X]).to_numpy(),
                area_definition(area_extent=area_extent),
            )
            for ds, area_extent in self.domain.iter_dataset_and_extent()
        ]

    def vstack(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nz, Ny, Nx, Nv | Nt], np.float_]:
        return np.stack(self._resample_point_over_time(longitude, latitude, time))  # (Z, Y, X, C*T)

    @property
    def plot(self) -> Callable[[float, float, Array[[N], np.datetime64]], PlotterAccessor]:
        return functools.partial(PlotterAccessor, self)


class PlotterAccessor(AbstractDomain):
    _index: dict[str, list[Any]]

    @property
    def domain(self) -> Domain:
        return self.resampler.domain

    def __init__(
        self, resampler: ReSampler, longitude: float, latitude: float, time: Array[[N], np.datetime64]
    ) -> None:
        super().__init__()
        self.resampler = resampler
        self.data = resampler(longitude, latitude, time)
        p_def = resampler._partial_area_definition(longitude, latitude)
        self.a_defs = [p_def(area_extent=area_extent) for area_extent in self.area_extents]
        self._index = {"time": time.tolist(), "levels": self.levels.tolist()}

    def index(self, key, value) -> int:
        return self._index[key].index(value)

    def level(self, time: np.datetime64, level: float) -> None:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from cartopy.mpl.geoaxes import GeoAxes
        from cartopy.feature import STATES

        tidx = self.index("time", time)
        lvl = self.index("levels", level)
        area_def = self.a_defs[lvl]

        z, t, q, u, v = self.data[:, tidx, lvl]  # 300 hPa
        x, y = area_def.get_lonlats()

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(1, 1, 1, projection=area_def.to_cartopy_crs())
        assert isinstance(ax, GeoAxes)

        ax.coastlines()
        transform = ccrs.PlateCarree()
        ax.add_feature(STATES)
        ax.gridlines()

        ax.contour(
            x,
            y,
            z,
            colors="k",
            linewidths=1,
            transform=transform,
        )

        ax.contour(
            x,
            y,
            t,
            colors="r",
            linewidths=1,
            linestyles="--",
            transform=transform,
        )
        ax.contourf(
            x,
            y,
            q,
            cmap="Greens",
            transform=transform,
        )

        ax.barbs(
            x,
            y,
            u,
            v,
            length=2,
            pivot="middle",
            transform=transform,
        )


def _get_resample_method(
    method: str,
    radius_of_influence=100_000,
    fill_value=0,
    reduce_data=True,
    nprocs=1,
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


def _area_definition(
    width: float,
    height: float,
    projection: pyproj.CRS | dict[str, Any],
    area_extent: Array[[N4], np.float_] | Sequence,
    lons: Array[[...], np.float_] | None = None,
    lats: Array[[...], np.float_] | None = None,
    dtype: Any = np.float_,
    area_id: str = "undefined",
    description: str = "undefined",
    proj_id: str = "undefined",
    nprocs: int = 1,
) -> pyresample.geometry.AreaDefinition:
    return pyresample.geometry.AreaDefinition(
        area_id,
        description,
        proj_id,
        width=width,
        height=height,
        projection=projection,
        area_extent=area_extent,
        lons=lons,
        lats=lats,
        dtype=dtype,
        nprocs=nprocs,
    )
