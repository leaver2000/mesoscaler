from __future__ import annotations

import datetime
import os

import numpy as np
import pytest
import xarray as xr

import src.mesoscaler as ms

here = os.path.abspath(os.path.dirname(__file__))
data = os.path.abspath(os.path.join(here, "data"))
urma_store = os.path.abspath(os.path.join(data, "urma.zarr"))
era5_store = os.path.abspath(os.path.join(data, "era5.zarr"))
assert os.path.exists(urma_store)
assert os.path.exists(era5_store)


@pytest.fixture
def data_paths() -> list[tuple[str, list[ms.URMA] | list[ms.ERA5]]]:
    items = {urma_store: ms.URMA.loc[["T2M", "U10", "V10"]], era5_store: ms.ERA5.loc[["U", "V", "T"]]}
    return list(ms.utils.chain_items(items))


@pytest.fixture()
def dataset_sequence(data_paths) -> ms.DatasetSequence:
    urma = []  # type: list[ms.DependentDataset]
    era5 = []  # type: list[ms.DependentDataset]
    for i in range(10):
        u, _ = ms.open_datasets(data_paths)
        u[ms.time] = u[ms.time].to_numpy() + np.timedelta64(datetime.timedelta(hours=i))

        urma.append(u)
    for i in range(10):
        i -= 5
        _, e = ms.open_datasets(data_paths)

        e[ms.time] = e[ms.time].to_numpy() + np.timedelta64(datetime.timedelta(hours=i))
        era5.append(e)

    return ms.DatasetSequence(
        [
            xr.concat(urma, dim=ms.time).chunk({ms.time: 1}),
            xr.concat(era5, dim=ms.time).chunk({ms.time: 1}),
        ]
    )
