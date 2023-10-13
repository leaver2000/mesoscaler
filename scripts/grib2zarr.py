"""
convert grib2 data to .zarr store
"""
import glob
import os

import tqdm
import xarray as xr
from src.mesoscaler.generic import CFDatasetEnum
from src.mesoscaler.datasets.urma.constants import (
    CEILING,
    DEWPOINT_TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    TOTAL_CLOUD_COVER,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    VISIBILITY,
    WIND_DIRECTION_10M,
    WIND_SPEED_10M,
    WIND_SPEED_GUST,
)

DEFAULT_VARIABLES = [
    TOTAL_CLOUD_COVER,
    CEILING,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    WIND_SPEED_10M,
    WIND_SPEED_GUST,
    WIND_DIRECTION_10M,
    TEMPERATURE_2M,
    DEWPOINT_TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    VISIBILITY,
]


def open_mfdataset(files: list[str], variables: list[CFDatasetEnum]):
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

    for dvar in variables:
        ds[dvar.value].attrs = dvar.md
    ds.attrs = {}
    return ds


def to_zarr(ds: xr.Dataset, out_path: str) -> None:
    if os.path.exists(out_path):
        ds.to_zarr(out_path, mode="a", consolidated=True, append_dim="time")
    else:
        ds.to_zarr(out_path, mode="w", consolidated=True)


def main(variables=DEFAULT_VARIABLES):
    data_path = "/mnt/data"
    folder_list = sorted(glob.glob(os.path.join(data_path, "urma2p5/*")))
    out_path = os.path.join(data_path, "urma2p5.zarr")

    for folder in tqdm.tqdm(folder_list, desc="GRIB->zarr", unit="day", total=len(folder_list)):
        grib_files = glob.glob(os.path.join(folder, "**grb2_wexp"))
        tqdm.tqdm.write(f"INFO: reading grib2 data from {folder}")

        try:
            ds = open_mfdataset(grib_files, variables)

            if os.path.exists(out_path):
                ds.to_zarr(out_path, mode="a", consolidated=True, append_dim="time")
            else:
                ds.to_zarr(out_path, mode="w", consolidated=True)
        except Exception as e:
            tqdm.tqdm.write(f"ERROR: {e}")


if __name__ == "__main__":
    main()
