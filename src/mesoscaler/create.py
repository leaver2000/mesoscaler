from __future__ import annotations

import datetime
import itertools

import numpy as np
import xarray as xr

from . import _compat, utils
from ._typing import (
    Any,
    Array,
    ArrayLike,
    Callable,
    ChainableItems,
    Hashable,
    HashableT,
    Iterable,
    ListLike,
    Mapping,
    Nt,
    Number,
    Nv,
    Nx,
    Ny,
    Nz,
    PointOverTime,
    StrPath,
    TypeAlias,
    overload,
)
from .core import (
    DEFAULT_DX,
    DEFAULT_HEIGHT,
    DEFAULT_LEVEL_START,
    DEFAULT_LEVEL_STEP,
    DEFAULT_LEVEL_STOP,
    DEFAULT_LEVELS,
    DEFAULT_PRESSURE_BASE,
    DEFAULT_PRESSURE_TOP,
    DEFAULT_RESAMPLE_METHOD,
    DEFAULT_SCALE_RATE,
    DEFAULT_WIDTH,
    STANDARD_SURFACE_PRESSURE,
    DataProducer,
    DependentDataset,
    Depends,
    Mesoscale,
    open_datasets,
)
from .enums import LAT, LON, LVL, TIME, Coordinates, DependentVariables, T, X, Y, Z
from .generic import DataGenerator
from .sampling.domain import DatasetSequence, Domain
from .sampling.resampler import ReSampler
from .sampling.sampler import LinearSampler

CoordinateValue: TypeAlias = (
    ListLike[float | np.datetime64 | datetime.datetime | str] | ArrayLike[np.float_ | np.datetime64]
)
DataValue: TypeAlias = ArrayLike | xr.DataArray | xr.Dataset | DependentDataset


def _open_datasets(
    datasets: Iterable[DependentDataset] | ChainableItems[StrPath, Depends], **kwargs: Any
) -> DatasetSequence:
    if isinstance(datasets, DatasetSequence):
        return datasets

    dsets = list(datasets)
    if not dsets:
        raise ValueError("datasets is empty")
    if all(isinstance(x, DependentDataset) for x in dsets):
        return DatasetSequence(dsets)  # type: ignore

    return open_datasets(dsets, **kwargs)  # type: ignore


@overload
def coordinates(**kwargs: CoordinateValue) -> xr.Coordinates:
    ...


@overload
def coordinates(__data: Mapping[HashableT, CoordinateValue], /) -> xr.Coordinates:
    ...


def coordinates(__data: Any | None = None, /, **kwargs: Any) -> xr.Coordinates:
    """
    This function will look for the aliases of the keyword arguments and remap it to the correct coordinate name.

    >>> create.coordinates(lon=[1, 2], lat=[1, 2], time=[datetime.datetime(2020, 1, 1)], level=[1013.25])
    Coordinates:
        longitude  (Y, X) float64 1.0 2.0 1.0 2.0
        latitude   (Y, X) float64 1.0 1.0 2.0 2.0
        time       (T) datetime64[ns] 2020-01-01
        vertical   (Z) float64 1.013e+03
    """

    data = {Coordinates(k): v for k, v in utils.chain_items(__data or kwargs)}

    lon, lat = (np.asanyarray(data[x]).astype(np.float_) for x in (LON, LAT))
    if lon.ndim != lat.ndim:
        raise ValueError("lon and lat must have the same number of dimensions")
    if lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    return xr.Coordinates(
        {
            LON: (LON.axis, lon),
            LAT: (LAT.axis, lat),
            TIME: (TIME.axis, np.asanyarray(data[TIME]).astype("datetime64[ns]")),
            LVL: (LVL.axis, np.asanyarray(data.get(LVL, [STANDARD_SURFACE_PRESSURE])).astype(np.float_)),
        }
    )


@overload
def data_array(
    __data: DataValue, __coordinates: xr.Coordinates | Mapping[HashableT, CoordinateValue], /
) -> xr.DataArray:
    ...


@overload
def data_array(__data: DataValue, /, **kwargs: CoordinateValue) -> xr.DataArray:
    ...


def data_array(__data: DataValue, __coordinates: Any = None, /, **kwargs: CoordinateValue) -> xr.DataArray:
    if __coordinates is None:
        __coordinates = coordinates(**kwargs)
    elif not isinstance(__coordinates, xr.Coordinates):
        __coordinates = coordinates(__coordinates)

    data = np.asanyarray(__data)
    if data.ndim == 3 and __coordinates[Z].size == 1:
        data = data[:, np.newaxis, ...]
    if data.ndim != 4:
        raise ValueError(f"Expected 3 or 4 dimensions, got {data.ndim}")

    return xr.DataArray(data, coords=__coordinates, dims=[T, Z, Y, X])


@overload
def dataset(
    __data: Mapping[DependentVariables, DataValue],
    __coordinates: xr.Coordinates | Mapping[Hashable, CoordinateValue],
    /,
) -> DependentDataset:
    ...


@overload
def dataset(__data: Mapping[DependentVariables, DataValue], /, **kwargs: CoordinateValue) -> DependentDataset:
    ...


def dataset(
    __data: Mapping[DependentVariables, DataValue], __coordinates: Any = None, /, **kwargs: CoordinateValue
) -> DependentDataset:
    if __coordinates is None:
        __coordinates = coordinates(**kwargs)
    elif not isinstance(__coordinates, xr.Coordinates):
        __coordinates = coordinates(__coordinates)

    return (
        DependentDataset(
            {key: data_array(data, __coordinates) for key, data in __data.items()}, depends=list(__data.keys())
        )
        .set_xindex(LVL)
        .set_xindex(TIME)
    )


def dataset_sequence(
    __data: Iterable[
        tuple[Mapping[DependentVariables, DataValue], xr.Coordinates | Mapping[Hashable, CoordinateValue]]
    ]
) -> DatasetSequence:
    return DatasetSequence(itertools.starmap(dataset, __data))


def resampler(
    dsets: Iterable[DependentDataset] | ChainableItems[StrPath, Depends],
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    *,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = DEFAULT_SCALE_RATE,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    method: str = DEFAULT_RESAMPLE_METHOD,
) -> ReSampler:
    dsets = _open_datasets(dsets, levels=levels)
    scale = Mesoscale.arange(dx, dy, start, stop, step, p0=p0, p1=p1, rate=rate, levels=levels)
    return scale.resample(dsets, height=height, width=width, method=method)


def producer(
    dsets: Iterable[DependentDataset] | ChainableItems[StrPath, Depends],
    indices: Iterable[PointOverTime] | Callable[[Domain], Iterable[PointOverTime]] = LinearSampler,
    *,
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = DEFAULT_SCALE_RATE,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    method: str = DEFAULT_RESAMPLE_METHOD,
    shuffle: bool = False,
    seed: int | None = None,
    **sampler_kwargs: Any,
) -> DataProducer:
    r = resampler(
        dsets,
        dx,
        dy,
        start,
        stop,
        step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
        height=height,
        width=width,
        method=method,
    )
    if callable(indices):
        indices = indices(r.domain, **sampler_kwargs)

    p = DataProducer(indices, resampler=r)
    if shuffle:
        p.shuffle(seed=seed or 0)
    return p


def generator(
    dsets: Iterable[DependentDataset] | ChainableItems[StrPath, Depends],
    indices: Iterable[PointOverTime] | Callable[[Domain], Iterable[PointOverTime]] = LinearSampler,
    *,
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = 12,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    method: str = DEFAULT_RESAMPLE_METHOD,
    # - data consumer -
    maxsize: int = 0,
    timeout: float | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    **sampler_kwargs: Any,
) -> DataGenerator[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]:
    data_producer = producer(
        dsets,
        indices,
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
        height=height,
        width=width,
        method=method,
        shuffle=shuffle,
        seed=seed,
        **sampler_kwargs,
    )
    return DataGenerator(data_producer, maxsize=maxsize, timeout=timeout)


def loader(
    dsets: Iterable[DependentDataset] | ChainableItems[StrPath, Depends],
    indices: Iterable[PointOverTime] | Callable[[Domain], Iterable[PointOverTime]] = LinearSampler,
    *,
    dx: float = DEFAULT_DX,
    dy: float | None = None,
    start: int = DEFAULT_LEVEL_START,
    stop: int = DEFAULT_LEVEL_STOP,
    step: int = DEFAULT_LEVEL_STEP,
    p0: float = DEFAULT_PRESSURE_BASE,
    p1: float = DEFAULT_PRESSURE_TOP,
    rate: float = 12,
    levels: ListLike[Number] = DEFAULT_LEVELS,
    height: int = 80,
    width: int = 80,
    method: str = "nearest",
    # - data consumer -
    maxsize: int = 0,
    timeout: float | None = None,
    # - data loader -
    shuffle: bool = False,
    seed: int | None = None,
    # - data loader -
    batch_size: int | None = 1,
    # shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **sampler_kwargs: Any,
) -> _compat.DataLoader[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]:
    if not _compat._has_torch:
        raise RuntimeError("torch is not installed!")
    data_generator = generator(
        dsets,
        indices,
        dx=dx,
        dy=dy,
        start=start,
        stop=stop,
        step=step,
        p0=p0,
        p1=p1,
        rate=rate,
        levels=levels,
        height=height,
        width=width,
        method=method,
        maxsize=maxsize,
        timeout=timeout,
        shuffle=shuffle,
        seed=seed,
        **sampler_kwargs,
    )
    return _compat.DataLoader(
        data_generator,
        batch_size=batch_size,
        timeout=timeout or 0,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
