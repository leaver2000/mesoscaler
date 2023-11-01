from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import zarr
from pyresample.geometry import AreaDefinition

from .. import _compat as compat, utils
from .._typing import (
    Any,
    Array,
    Callable,
    Latitude,
    Literal,
    Longitude,
    N,
    Nt,
    Nv,
    Nx,
    Ny,
    Nz,
)
from ..time64 import DateTimeValue


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

    def _assert_dim(self, f: Callable[[Any], bool]) -> None:
        if not f(self.dim):
            raise ValueError(f"dim {self.dim} is not valid!")


class PlotArray:
    def __init__(
        self,
        array: Array[[Nv, Nt, Nz, Ny, Nx], np.float_],
        longitude: Longitude,
        latitude: Latitude,
        time: Array[[N], np.datetime64],
        levels: Array[[N], np.float_],
        width: int,
        height: int,
        area_definitions: list[AreaDefinition],
        transform: compat.ccrs.Projection | None = None,
        features: list = [],
        grid_lines: bool = True,
        coast_lines: bool = True,
    ) -> None:
        if not compat.has_cartopy:
            raise ImportError("cartopy is not installed")
        super().__init__()
        self._plot_options = {"contour": self._contour, "barbs": self._barbs, "contourf": self._contourf}
        self._array = array
        self._center = (longitude, latitude)
        self._width = width
        self._height = height
        self._time = time
        self._area_defs = area_definitions
        self._levels = levels
        self._config = {
            "grid_lines": grid_lines,
            "coast_lines": coast_lines,
            "features": features,
            "transform": transform or compat.ccrs.PlateCarree(),
        }
        self._fig = None

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self._levels

    @property
    def array(self) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        return self._array

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self._array.shape  # type: ignore

    @property
    def features(self) -> list:
        return self._config["features"]

    @property
    def transform(self) -> compat.ccrs.Projection:
        return self._config["transform"]

    @property
    def grid_lines(self) -> bool:
        return self._config["grid_lines"]

    @property
    def coast_lines(self) -> bool:
        return self._config["coast_lines"]

    # =================================================================================================================
    def gcf(self, figsize=(10, 10)):
        if self._fig is None:
            self._fig = compat.plt.figure(figsize=figsize)
        return self._fig

    def level(
        self,
        tidx: DateTimeValue | int,
        zidx: int,
        options: list[PlotOption],
        ax: compat.GeoAxes,
        show_inner_domains: bool = True,
    ) -> None:
        if not isinstance(tidx, int):
            condition = self._time == np.datetime64(tidx)
            if not condition.any():
                raise ValueError(f"tidx {tidx} is not in the domain")
            tidx = int(np.nonzero(condition)[0][0])

        area_def = self._area_defs[zidx]
        x, y = area_def.get_lonlats()

        if show_inner_domains:
            for ad in self._area_defs[:zidx]:
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
            lvl = self.levels[zidx]
            ax.set_title(f"Level: {lvl}")
        for opt in options:
            self._plot_options[opt.kind](ax, x, y, tidx, zidx, opt)

    def all_levels(
        self,
        tidx: DateTimeValue | int,
        options: list[PlotOption],
        size: int = 10,
        show_inner_domains: bool = True,
    ) -> None:
        ratio = self._width / self._height
        Z = len(self.levels)
        fig = self.gcf(figsize=(size, (size / ratio) * Z))
        fig.tight_layout()

        it = [
            (z, fig.add_subplot(Z, 1, Z - z, projection=area_def.to_cartopy_crs()))
            for z, area_def in enumerate(self._area_defs)
        ]
        for zidx, ax in it:
            self.level(tidx, zidx, ax=ax, options=options, show_inner_domains=show_inner_domains)  # type: ignore

    # =================================================================================================================
    def _barbs(
        self,
        ax: compat.GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        tidx: int,
        zidx: int,
        opt: PlotOption,
    ) -> None:
        opt._assert_dim(utils.is_integer_pair)
        u, v = self.array[opt.dim, tidx, zidx, :, :]
        ax.barbs(x, y, u, v, length=4, transform=self.transform, sizes=dict(emptybarb=0.01), alpha=opt.alpha)

    def _contour(
        self,
        ax: compat.GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        tidx: int,
        zidx: int,
        opt: PlotOption,
    ) -> None:
        opt._assert_dim(utils.is_integer)
        z = self.array[opt.dim, tidx, zidx, :, :]
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
        ax: compat.GeoAxes,
        x: Array[[Ny, Nx], np.float_],
        y: Array[[Ny, Nx], np.float_],
        tidx: int,
        zidx: int,
        opt: PlotOption,
    ) -> None:
        opt._assert_dim(utils.is_integer)
        z = self.array[opt.dim, tidx, zidx, :, :]
        ax.contourf(x, y, z, transform=self.transform, alpha=opt.alpha, cmap=opt.cmap)

    def _plot_inner_domain(self, ax: compat.GeoAxes, x: Array, y: Array) -> None:
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

    @classmethod
    def from_group(
        cls,
        g: zarr.Array | zarr.Group,
        sample_idx: int,
        projection: compat.ccrs.Projection | None = None,
        features: list = [],
        grid_lines: bool = True,
        coast_lines: bool = True,
    ) -> PlotArray:
        import functools

        array = g[sample_idx, ...]

        width = g.attrs["width"]
        height = g.attrs["height"]
        proj = g.attrs["proj"]

        md = g.attrs["metadata"][sample_idx]
        longitude = md["longitude"]
        latitude = md["latitude"]
        pdef = functools.partial(
            utils.area_definition,
            width=width,
            height=height,
            projection={"proj": proj, "lon_0": longitude, "lat_0": latitude},
        )
        scaling = sorted(g.attrs["scaling"], key=lambda x: x["level"], reverse=True)
        extent = np.array([x["extent"] for x in scaling]) * 1000
        return PlotArray(
            array=array,  # type: ignore
            longitude=longitude,
            latitude=latitude,
            time=md["time"],
            levels=np.array([x["level"] for x in scaling]),
            width=g.attrs["width"],
            height=g.attrs["height"],
            area_definitions=[pdef(area_extent=ext) for ext in extent],
            transform=projection,
            features=features,
            grid_lines=grid_lines,
            coast_lines=coast_lines,
        )
