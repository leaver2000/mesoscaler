from __future__ import annotations

import os
import datetime
import xarray as xr
import numpy as np
import pytest


import src.mesoscaler as mesoscaler

TIME, LEVEL, LAT, LON = mesoscaler.unpack_coords()

here = os.path.abspath(os.path.dirname(__file__))
data = os.path.abspath(os.path.join(here, "data"))
urma_store = os.path.abspath(os.path.join(data, "urma.zarr"))
era5_store = os.path.abspath(os.path.join(data, "era5.zarr"))
assert os.path.exists(urma_store)
assert os.path.exists(era5_store)


@pytest.fixture
def data_paths() -> list[tuple[str, list[mesoscaler.URMA] | list[mesoscaler.ERA5]]]:
    items = {urma_store: mesoscaler.URMA.loc[["T2M", "U10", "V10"]], era5_store: mesoscaler.ERA5.loc[["U", "V", "T"]]}  # type: ignore
    return list(mesoscaler.utils.items(items))


@pytest.fixture
def dataset_sequence(data_paths) -> mesoscaler.DatasetSequence:
    urma = []
    era5 = []
    for i in range(10):
        u, _ = mesoscaler.open_datasets(data_paths)
        u[TIME] = u[TIME].to_numpy() + np.timedelta64(datetime.timedelta(hours=i))

        urma.append(u)
    for i in range(10):
        i -= 5
        _, e = mesoscaler.open_datasets(data_paths)

        e[TIME] = e[TIME].to_numpy() + np.timedelta64(datetime.timedelta(hours=i))
        era5.append(e)

    return mesoscaler.DatasetSequence([xr.concat(urma, dim=TIME), xr.concat(era5, dim=TIME)])
