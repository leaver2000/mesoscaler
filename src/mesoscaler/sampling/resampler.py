from __future__ import annotations

import functools

import numpy as np
import pyresample.geometry
from pyresample.geometry import AreaDefinition, GridDefinition

from .._typing import (
    Any,
    Array,
    Callable,
    Iterator,
    Latitude,
    Longitude,
    N,
    Nt,
    Nv,
    Nx,
    Ny,
    Nz,
    TimeSlice,
)
from ..enums import TIME, CoordinateReferenceSystem, LiteralCRS, X, Y
from ..utils import area_definition
from .intersection import AbstractDomain, DatasetAndExtent, DomainIntersection


def _get_resample_method(
    method: str,
    radius_of_influence=500000,
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


class ReSampler(AbstractDomain):
    # There are alot of callbacks and partial methods in this class.
    @property
    def domain(self) -> DomainIntersection:
        return self._domain

    def __init__(
        self,
        domain: DomainIntersection,
        /,
        *,
        height: int = 80,
        width: int = 80,
        target_projection: LiteralCRS = "lambert_azimuthal_equal_area",
        method: str = "nearest",
        sigmas=[1.0],
        radius_of_influence: int = 500000,
        fill_value: int = 0,
        reduce_data: bool = True,
        nprocs: int = 1,
        segments: Any = None,
        with_uncert: bool = False,
    ) -> None:
        super().__init__()
        self._domain = domain
        self.height = height
        self.width = width
        self.target_projection = (
            CoordinateReferenceSystem[target_projection] if isinstance(target_projection, str) else target_projection
        )

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
            area_definition,
            width=self.width,
            height=self.height,
            projection=self.target_projection.from_point(longitude, latitude),
        )

    def _resample_point_over_time(
        self, longitude: Longitude, latitude: Latitude, time: TimeSlice | Array[[N], np.datetime64]
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

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: TimeSlice | Array[[N], np.datetime64]
    ) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        """stack the data along `Nz` and reshape and unsqueeze the data to match the expected output."""
        arr = np.stack(self._resample_point_over_time(longitude, latitude, time))  # (z, y, x, v*t)

        # - reshape the data
        # t = len(self.slice_time(time))
        z, y, x = arr.shape[:3]
        arr = arr.reshape((z, y, x, -1, self.n_vars))  # unsqueeze C
        return np.moveaxis(arr, (-1, -2), (0, 1))

    def iter_dataset_and_extent(self) -> Iterator[DatasetAndExtent]:
        return zip(self.datasets, self.area_extents)
