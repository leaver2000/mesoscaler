from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import xarray as xr

try:
    import gcsfs  # noqa: F401
except ImportError:
    raise ImportError("Please install gcsfs to use this script. (pip install gcsfs)")
# https://discuss.pytorch.org/t/dataloader-parallelization-synchronization-with-zarr-xarray-dask/176149

import mesoscaler as ms
from mesoscaler.enums import (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
)

GOOGLE_STORAGE = "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
DEFAULT_VARIABLES = [
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    # VERTICAL_VELOCITY,
]


def main(
    *,
    local_directory: str,
    start_date: datetime.date | datetime.datetime | np.datetime64 | str,
    end_date: datetime.date | datetime.datetime | np.datetime64 | str,
    data_variables: list[ms.ERA5] = DEFAULT_VARIABLES,
) -> int:
    time = np.s_[ms.days.datetime(start_date) : ms.days.datetime(end_date) + ms.days.delta(1)]
    ds = xr.open_zarr(GOOGLE_STORAGE).sel(time=time)
    ds = ds[data_variables].sel(level=ds.level >= 200)
    ds.to_zarr(local_directory, mode="w")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data from Google Cloud Storage.")
    parser.add_argument(
        "local_directory",
        metavar="local_directory",
        type=str,
        help="Path to the local store where the data will be saved.",
    )
    args = parser.parse_args()

    sys.exit(main(**vars(args)))
