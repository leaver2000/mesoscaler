import sys
import datetime
import xarray as xr
import numpy as np

# https://discuss.pytorch.org/t/dataloader-parallelization-synchronization-with-zarr-xarray-dask/176149
GEOPOTENTIAL = "geopotential"
SPECIFIC_HUMIDITY = "specific_humidity"
TEMPERATURE = "temperature"
U_COMPONENT_OF_WIND = "u_component_of_wind"
V_COMPONENT_OF_WIND = "v_component_of_wind"
VERTICAL_VELOCITY = "vertical_velocity"

UPPER_AIR_VARIABLES = [
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    VERTICAL_VELOCITY,
]


def main(
    google_store="gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
    local_store: str = "/mnt/data/era5/2019-2022-upper-air-1h-0p25deg.zarr",
) -> int:
    ds = xr.open_zarr(google_store).sel(time=np.s_[datetime.datetime(2019, 1, 1) :])
    ds = ds[UPPER_AIR_VARIABLES].sel(level=ds.level >= 200)
    ds.to_zarr(local_store)

    return 0


if __name__ == "__main__":
    sys.exit(main())
