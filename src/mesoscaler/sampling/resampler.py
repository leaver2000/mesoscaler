from __future__ import annotations

import abc
import functools

import numpy as np
import pyresample.geometry
from pyresample.geometry import AreaDefinition, GridDefinition

from .._typing import (
    TYPE_CHECKING,
    Any,
    AreaExtent,
    Array,
    Callable,
    Iterable,
    Iterator,
    Latitude,
    Longitude,
    N,
    TimeSlice,
    TypeAlias,
)
from ..enums import LVL, TIME, CoordinateReferenceSystem, LiteralCRS, X, Y
from ..utils import area_definition, slice_time

if TYPE_CHECKING:
    from ..core import DependentDataset, Mesoscale, Nt, Nv, Nx, Ny, Nz

ResampleInstruction: TypeAlias = tuple["DependentDataset", AreaExtent]


class AbstractInstructor(abc.ABC):
    @property
    @abc.abstractmethod
    def instructor(self) -> ReSampleInstructor:
        ...

    @property
    def dvars(self) -> Array[[N, N], np.str_]:
        return self.instructor._dvars

    @property
    def levels(self) -> Array[[N], np.float_]:
        return self.instructor._levels

    @property
    def time(self) -> Array[[N], np.datetime64]:
        return self.instructor._time

    def slice_time(self, s: TimeSlice, /) -> Array[[N], np.datetime64]:
        return slice_time(self.time, s)


class ReSampleInstructor(Iterable[ResampleInstruction], AbstractInstructor):
    def __init__(self, scale: Mesoscale, *dsets: DependentDataset) -> None:
        super().__init__()

        time, levels, dvars = zip(*((ds.time.to_numpy(), ds.level.to_numpy(), list(ds.data_vars)) for ds in dsets))
        # - time
        self._time = time = np.sort(np.unique(time))

        # - levels
        levels = np.concatenate(levels)
        self._levels = levels = np.sort(levels[np.isin(levels, scale.hpa)])[::-1]  # descending

        # - dvars
        self._dvars = np.stack(dvars)

        datasets = (ds.sel({LVL: [lvl], TIME: time}) for lvl in levels for ds in dsets if lvl in ds.level)
        extents = scale.stack_extent() * 1000.0  # km -> m
        self._it = tuple(zip(datasets, extents))

    def __iter__(self) -> Iterator[ResampleInstruction]:
        return iter(self._it)

    @property
    def instructor(self) -> ReSampleInstructor:
        return self


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


class ReSampler(AbstractInstructor):
    # There are alot of callbacks and partial methods in this class.
    @property
    def instructor(self) -> ReSampleInstructor:
        return self._instructor

    def __init__(
        self,
        instructor: ReSampleInstructor,
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
        self._instructor = instructor
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
        self, longitude: Longitude, latitude: Latitude, time: TimeSlice
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
            for ds, area_extent in self._instructor
        ]

    def __call__(
        self, longitude: Longitude, latitude: Latitude, time: TimeSlice
    ) -> Array[[Nv, Nt, Nz, Ny, Nx], np.float_]:
        """stack the data along `Nz` and reshape and unsqueeze the data to match the expected output."""
        arr = np.stack(self._resample_point_over_time(longitude, latitude, time))  # (z, y, x, v*t)

        # - reshape the data
        t = len(self.slice_time(time))
        z, y, x = arr.shape[:3]
        arr = arr.reshape((z, y, x, t, -1))  # unsqueeze C
        return np.moveaxis(arr, (-1, -2), (0, 1))
