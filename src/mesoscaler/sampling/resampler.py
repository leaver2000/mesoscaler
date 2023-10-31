from __future__ import annotations

import abc
import dataclasses
import functools

import numpy as np
import pyproj
import pyresample.geometry
import xarray as xr
from pyresample.geometry import AreaDefinition, GridDefinition

from .._typing import (
    N4,
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes
try:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes

    has_cartopy = True
except ImportError:
    has_cartopy = False


_VARIABLES = "variables"


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
        """stack the data along `Nz` -> unsqueeze the variables & time -> reshape the data to match the expected output."""

        arr = self.resampler.zstack(longitude, latitude, time)  # (Z, Y, X, C*T)
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

    @property
    def resampler(self) -> ReSampler:
        return self

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

    def _partial_area_definition(self, longitude: Longitude, latitude: Latitude) -> functools.partial[AreaDefinition]:
        return functools.partial(
            _area_definition,
            width=self.width,
            height=self.height,
            projection={"proj": self.proj, "lon_0": longitude, "lat_0": latitude},
        )

    # -
    def zstack(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64]
    ) -> Array[[Nz, Ny, Nx, Nv | Nt], np.float_]:
        return np.stack(self._resample_point_over_time(longitude, latitude, time))  # (Z, Y, X, C*T)

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
            for ds, area_extent in self.iter_dataset_and_extent()
        ]

    @property
    def plot(self) -> functools.partial[SamplePlotter]:
        return functools.partial(SamplePlotter, self)


@dataclasses.dataclass
class PlotOption:
    name: str
    kind: Literal["contour", "barbs", "contourf"]
    dim: int | tuple[int, int]
    colors: str = "r"
    linestyles: str = "--"
    linewidths: float = 0.75
    cmap: str = "Greens"
    alpha: float = 1.0


class SamplePlotter(AbstractResampler):
    @property
    def resampler(self) -> ReSampler:
        return self._resampler

    def __init__(
        self,
        resampler: ReSampler,
        longitude: Longitude,
        latitude: Latitude,
        time: Array[[N], np.datetime64],
        /,
        *,
        transform: ccrs.Projection | None = None,
        features: list = [],
        grid_lines: bool = True,
        coast_lines: bool = True,
    ) -> None:
        if not has_cartopy:
            raise ImportError("cartopy is not installed")
        super().__init__()
        self._resampler = resampler
        self.sample = sample = self.resampler(longitude, latitude, time)
        self.shape = self.NV, self.NT, self.NZ, self.NY, self.NX = sample.shape
        self.center = (longitude, latitude)
        self._area_defs = [
            _area_definition(
                self.width,
                self.height,
                {"proj": self.proj, "lon_0": longitude, "lat_0": latitude, "units": "m"},
                area_extent=self.area_extent[z],
            )
            for z in range(self.NZ)
        ]

        self._plot_options = {"contour": self._contour, "barbs": self._barbs, "contourf": self._contourf}
        self._config = {
            "grid_lines": grid_lines,
            "coast_lines": coast_lines,
            "transform": transform or ccrs.PlateCarree(),
            "features": features,
        }
        self._fig = None

    @property
    def features(self) -> list:
        return self._config["features"]

    @property
    def transform(self) -> ccrs.Projection:
        return self._config["transform"]

    @property
    def grid_lines(self) -> bool:
        return self._config["grid_lines"]

    @property
    def coast_lines(self) -> bool:
        return self._config["coast_lines"]

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: Array[[N], np.datetime64], **kwargs
    ) -> SamplePlotter:
        """Create a new BatchPlotter with the same resampler and batch data, but with a different center."""
        return self.resampler.plot(longitude, latitude, time, **(self._config | kwargs))

    # =================================================================================================================
    def gcf(self, figsize=(10, 10)):
        if self._fig is None:
            self._fig = plt.figure(figsize=figsize)
        return self._fig

    def level(
        self, t_dim: int, z_dim: int, options: list[PlotOption], ax: GeoAxes, show_inner_domains: bool = True
    ) -> None:
        area_def = self._area_defs[z_dim]
        x, y = area_def.get_lonlats()

        if show_inner_domains:
            for ad in self._area_defs[:z_dim]:
                self._plot_inner_domain(ax, *ad.get_lonlats())

        if self.grid_lines:
            ax.gridlines()
        if self.coast_lines:
            ax.coastlines()

        for feature in self.features:
            ax.add_feature(feature)

        ax.plot(
            x,
            y,
            marker="x",
            color="r",
            linewidth=1,
        )

        if self.levels is not None:
            lvl = self.levels[z_dim]
            ax.set_title(f"Level: {lvl}")
        for opt in options:
            self._plot_options[opt.kind](ax, x, y, t_dim, z_dim, opt)

    def all_levels(
        self, t_dim: int, options: list[PlotOption], size: int = 10, show_inner_domains: bool = True
    ) -> None:
        ratio = self.width / self.height
        fig = self.gcf(figsize=(size, (size / ratio) * self.NZ))
        fig.tight_layout()

        it = [
            (z, fig.add_subplot(self.NZ, 1, self.NZ - z, projection=area_def.to_cartopy_crs()))
            for z, area_def in enumerate(self._area_defs)
        ]
        for z_dim, ax in it:
            self.level(t_dim, z_dim, ax=ax, options=options, show_inner_domains=show_inner_domains)  # type: ignore

    # =================================================================================================================
    def _barbs(
        self,
        ax: GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        t_dim: int,
        z_dim: int,
        opt: PlotOption,
    ) -> None:
        u, v = self.sample[opt.dim, t_dim, z_dim, :, :]
        ax.barbs(x, y, u, v, length=4, transform=self.transform, sizes=dict(emptybarb=0.01), alpha=opt.alpha)

    def _contour(
        self,
        ax: GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        t_dim: int,
        z_dim: int,
        opt: PlotOption,
    ) -> None:
        z = self.sample[opt.dim, t_dim, z_dim, :, :]
        ax.contour(
            x,
            y,
            z,
            transform=self.transform,
            colors=opt.colors,
            alpha=opt.alpha,
            linewidths=opt.linewidths,
            linestyles=opt.linestyles,
        )

    def _contourf(
        self,
        ax: GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        t_dim: int,
        z_dim: int,
        opt: PlotOption,
    ) -> None:
        z = self.sample[opt.dim, t_dim, z_dim, :, :]
        ax.contourf(x, y, z, transform=self.transform, alpha=opt.alpha, cmap=opt.cmap)

    def _plot_inner_domain(self, ax: GeoAxes, x: Array, y: Array) -> None:
        x0, y0 = x[0, 0], y[0, 0]
        x1, y1 = x[-1, -1], y[-1, -1]
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color="purple",
            linewidth=1,
            transform=self.transform,
            linestyle="-.",
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
