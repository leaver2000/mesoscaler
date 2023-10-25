import sys
import os
import datetime
from typing import Any, Iterable
import s3fs
import glob
import xarray as xr
from mesoscaler.enums import URMA
from mesoscaler.enums import (
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    SURFACE_PRESSURE,
)

aws_bucket = "s3://noaa-urma-pds/urma2p5.*"
URMA2P5_DATE_FMT = "noaa-urma-pds/urma2p5.%Y%m%d"
client = s3fs.S3FileSystem(anon=True)


def filter_dates(
    sources: Iterable[tuple[Any, datetime.datetime]], start_date: datetime.datetime, end_date: datetime.datetime
) -> Iterable[tuple[Any, datetime.datetime]]:
    return filter(lambda x: x[1] >= start_date and x[1] <= end_date, sources)


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


def main(
    start_date: datetime.datetime = datetime.datetime(2019, 1, 1),
    end_date: datetime.datetime = datetime.datetime(2022, 1, 1),
    grib_folder: str = "/mnt/data/urma-gribs",
    zarr_store: str = "/mnt/data/urma2p5.zarr",
    data_variables=[
        SURFACE_PRESSURE,
        TEMPERATURE_2M,
        SPECIFIC_HUMIDITY_2M,
        U_WIND_COMPONENT_10M,
        V_WIND_COMPONENT_10M,
        SURFACE_PRESSURE,
    ],
) -> int:
    if not os.path.exists(grib_folder):
        os.makedirs(grib_folder)
    # - download grib data -
    if not glob.glob(os.path.join(grib_folder, "*.grb2_wexp")):  # dont download if already downloaded
        it = filter_dates(
            ((x, datetime.datetime.strptime(str(x), URMA2P5_DATE_FMT)) for x in client.glob(aws_bucket)),
            start_date,
            end_date,
        )
        for file, date in it:
            print(f"Downloading {file} {date}")
            url = client.glob(f"s3://{file}/urma2p5.t*2dvaranl_ndfd.grb2_wexp")
            client.get(url, grib_folder)
    # - convert to zarr -
    ds = open_mfdataset(
        glob.glob(os.path.join(grib_folder, "*.grb2_wexp")),
        data_variables,
    )
    ds.to_zarr(zarr_store, mode="w")
    return 0


if __name__ == "__main__":
    sys.exit(main())
