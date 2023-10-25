from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
from typing import Any, Iterable

import s3fs
import xarray as xr
import numpy as np

import mesoscaler as ms
from mesoscaler.enums import (
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    U_WIND_COMPONENT_10M,
    URMA,
    V_WIND_COMPONENT_10M,
)

AWS_S3_BUCKET = "s3://noaa-urma-pds/urma2p5.*"
URMA2P5_DATE_FMT = "noaa-urma-pds/urma2p5.%Y%m%d"


def open_mfdataset(files: list[str], variables: list[URMA]):
    """
    Wraps the xr.open_mfdataset function to filter by type of level and level.
    This is necessary for many of the NDFD datasets, which use inconsistent
    naming schemes for the `level` coordinate.
    """

    fsets = {(dvar.type_of_level, dvar.level) for dvar in variables}

    dsets = [
        xr.open_mfdataset(
            files,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            filter_by_keys={"typeOfLevel": tol, "level": lvl, "step": 0},
        ).drop_vars([tol, "step", "valid_time"], errors="ignore")
        for tol, lvl in fsets
    ]
    ds = xr.merge(dsets).rename({dvar.short_name: dvar for dvar in variables})[variables]

    ds.attrs = {}
    return ds


def filter_dates(
    sources: Iterable[tuple[Any, datetime.datetime]], start_date: np.datetime64, end_date: np.datetime64
) -> Iterable[tuple[Any, datetime.datetime]]:
    return filter(lambda x: x[1] >= start_date and x[1] <= end_date, sources)


def main(
    local_directory: str,
    start_date: datetime.datetime | np.datetime64 | str = "2019-01-01",
    end_date: datetime.datetime | np.datetime64 | str = "2022-01-01",
    grib_folder: str = "/mnt/data/urma-gribs",
    data_variables=[
        SURFACE_PRESSURE,
        TEMPERATURE_2M,
        SPECIFIC_HUMIDITY_2M,
        U_WIND_COMPONENT_10M,
        V_WIND_COMPONENT_10M,
        SURFACE_PRESSURE,
    ],
) -> int:
    client = s3fs.S3FileSystem(anon=True)

    start_date, end_date = ms.days.datetime(start_date), ms.days.datetime(end_date)
    if not os.path.exists(grib_folder):
        os.makedirs(grib_folder)

    # - download grib data -
    # if not glob.glob(os.path.join(grib_folder, "*.grb2_wexp")):  # dont download if already downloaded
    file_dates = filter_dates(
        ((x, datetime.datetime.strptime(str(x), URMA2P5_DATE_FMT)) for x in client.glob(AWS_S3_BUCKET)),
        start_date,
        end_date,
    )
    for file, date in file_dates:
        print(f"Downloading {file} {date}")
        url = client.glob(f"s3://{file}/urma2p5.t*2dvaranl_ndfd.grb2_wexp")
        client.get(url, grib_folder)

    # - convert to zarr -
    ds = open_mfdataset(
        glob.glob(os.path.join(grib_folder, "*.grb2_wexp")),
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
    args = parser.parse_args()

    sys.exit(main(args.local_directory))
