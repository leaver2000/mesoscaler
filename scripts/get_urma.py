from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
from typing import Any, Iterable

import numpy as np
import s3fs
import xarray as xr

import mesoscaler as ms
from mesoscaler.enums import (
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
)

AWS_S3_BUCKET = "s3://noaa-urma-pds/urma2p5.*"
URMA2P5_DATE_FMT = "noaa-urma-pds/urma2p5.%Y%m%d"
DEFAULT_URMA_VARIABLES = [
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    SURFACE_PRESSURE,
]


def open_mfdataset(files: list[str], variables: list[ms.URMA]) -> xr.Dataset:
    """
    Wraps the xr.open_mfdataset function to filter by type of level and level.
    This is necessary for many of the NDFD datasets because of how levels are
    represented.
    """

    fsets = {(dvar.type_of_level, dvar.level) for dvar in variables}

    dsets = [
        xr.open_mfdataset(
            files,
            chunks={},
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            filter_by_keys={"typeOfLevel": tol, "level": lvl, "step": 0},
        ).drop_vars([tol, "step", "valid_time"], errors="ignore")
        for tol, lvl in fsets
    ]
    ds = xr.merge(dsets).rename({dvar.short_name: dvar for dvar in variables})
    ds = ds[variables]
    ds.attrs = {}
    return ds


def filter_by_dates(sources: Iterable[Any], start_date: np.datetime64, end_date: np.datetime64) -> Iterable[str]:
    for source in map(str, sources):
        time = datetime.datetime.strptime(source, URMA2P5_DATE_FMT)
        if time >= start_date and time <= end_date:
            yield source


def main(
    *,
    local_directory: str,
    start_date: datetime.datetime | np.datetime64 | str,
    end_date: datetime.datetime | np.datetime64 | str,
    temp_grib_dir: str,
    data_variables: list[ms.URMA] = DEFAULT_URMA_VARIABLES,
) -> int:
    client = s3fs.S3FileSystem(anon=True)

    start_date, end_date = ms.days.datetime(start_date), ms.days.datetime(end_date)
    if not os.path.exists(temp_grib_dir):
        os.makedirs(temp_grib_dir)

    for directory in filter_by_dates(client.glob(AWS_S3_BUCKET), start_date, end_date):
        print(f"Downloading {directory}")
        resources = client.glob(f"s3://{directory}/urma2p5.t*2dvaranl_ndfd.grb2_wexp")
        client.get(resources, temp_grib_dir)

    # - convert to zarr -
    ds = open_mfdataset(
        glob.glob(os.path.join(temp_grib_dir, "*.grb2_wexp")),
        data_variables,
    )
    ds.to_zarr(local_directory, mode="w")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URMA data from Google Cloud Storage.")
    parser.add_argument(
        "local_directory",
        metavar="local_directory",
        type=str,
        help="Path to the local store where the data will be saved.",
    )
    parser.add_argument(
        "--start-date",
        metavar="start_date",
        type=str,
        default="2019-01-01",
        help="Start date for the data to download.",
    )
    parser.add_argument(
        "--end-date",
        metavar="end_date",
        type=str,
        default="2022-01-01",
        help="End date for the data to download.",
    )
    parser.add_argument(
        "--temp-grib-dir",
        metavar="temp_grib_dir",
        type=str,
        default="/tmp/grib",
        help="Temporary directory to store grib files.",
    )

    args = parser.parse_args()

    sys.exit(main(**vars(args)))
