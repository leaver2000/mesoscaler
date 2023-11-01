from __future__ import annotations

import os
import sys

import tqdm
import zarr

import mesoscaler as ms
from src.mesoscaler.enums import (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE,
    TEMPERATURE_2M,
    U_COMPONENT_OF_WIND,
    U_WIND_COMPONENT_10M,
    V_COMPONENT_OF_WIND,
    V_WIND_COMPONENT_10M,
)

era5_dvars = [
    GEOPOTENTIAL,
    TEMPERATURE,
    SPECIFIC_HUMIDITY,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
]

urma_dvars = [
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
]
_local_data = os.path.abspath("data")

urma_store = os.path.join(_local_data, "urma.zarr")
assert os.path.exists(urma_store)
era5_store = os.path.join(_local_data, "era5.zarr")
assert os.path.exists(era5_store)


def main(store: str = "data.zarr") -> int:
    dataset_sequence = ms.open_datasets([(urma_store, urma_dvars), (era5_store, era5_dvars)])
    # slight buffer around the bounds of the URMA dataset
    # -138. 57. -59. 19.
    area_of_interest = -120.0, 55.0, -65.0, 25.0

    ratio = 2 / 1
    height = 40  # px
    dy = 100  # km
    dx = int(dy * ratio)  # km
    width = int(height * ratio)  # px
    levels = [1013.25, 1000, 925, 850, 700, 600, 500, 400, 300, 200]
    scale = ms.Mesoscale(dx, dy, levels=levels)
    resampler = scale.create_resampler(dataset_sequence, width=width, height=height)
    sampler = ms.AreaOfInterestSampler(resampler.domain, aoi=area_of_interest, num_time=3, lon_lat_step=10)

    print(f"preparing to resample {len(sampler)} images and write to {store}")
    arrays = [resampler(lon, lat, time) for (lon, lat), time in tqdm.tqdm(sampler)]

    zarr.save_array(store, arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
