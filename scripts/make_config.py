import dataclasses
import datetime
from typing import Any, Hashable

from src.mesoscaler.generic import BaseConfig


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    source: str = ""
    module: str = ""

    start_date: datetime.datetime = datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc)
    end_date: datetime.datetime = datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc)

    references: list[str] = dataclasses.field(default_factory=list)
    channels: list[dict[Hashable, Any]] = dataclasses.field(default_factory=list)
    crs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class APIConfig(BaseConfig):
    datasets: list[DatasetConfig]


def main():
    urma = DatasetConfig(
        "URMAA 2.5km CONUS",
        "Unrestricted Mesoscale Analysis (URMA) 2.5km CONUS",
        "NOAA Real-Time Mesoscale Analysis (RTMA) / Unrestricted Mesoscale Analysis (URMA) was accessed on DATE from https://registry.opendata.aws/noaa-rtma.",
        "mesoscaler.datasets.urma",
        datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc),
        [
            "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/rtma.php",
            "https://graphical.weather.gov/docs/ndfdSRS.htm",
        ],
        channels=[
            {
                "description": "",
                "grib_id": 228164,
                "short_name": "tcc",
                "cf_var": "tcc",
                "long_name": "total_cloud_cover",
                "type_of_level": "atmosphereSingleLayer",
                "level": 0,
                "units": "%",
            },
            {
                "description": "",
                "grib_id": 260109,
                "short_name": "ceil",
                "cf_var": "ceil",
                "long_name": "ceiling",
                "type_of_level": "cloudCeiling",
                "level": 0,
                "units": "m",
            },
            {
                "description": "",
                "grib_id": 165,
                "short_name": "10u",
                "cf_var": "u10",
                "long_name": "10m_u_wind_component",
                "type_of_level": "heightAboveGround",
                "level": 10,
                "units": "m s**-1",
            },
            {
                "description": "",
                "grib_id": 166,
                "short_name": "10v",
                "cf_var": "v10",
                "long_name": "10m_v_wind_component",
                "type_of_level": "heightAboveGround",
                "level": 10,
                "units": "m s**-1",
            },
            {
                "description": "",
                "grib_id": 207,
                "short_name": "10si",
                "cf_var": "si10",
                "long_name": "10m_wind_speed",
                "type_of_level": "heightAboveGround",
                "level": 10,
                "units": "m s**-1",
            },
            {
                "description": "",
                "grib_id": 260065,
                "short_name": "gust",
                "cf_var": "gust",
                "long_name": "wind_speed_gust",
                "type_of_level": "heightAboveGround",
                "level": 0,
                "units": "m s**-1",
            },
            {
                "description": "",
                "grib_id": 260260,
                "short_name": "10wdir",
                "cf_var": "wdir10",
                "long_name": "10m_wind_direction",
                "type_of_level": "heightAboveGround",
                "level": 10,
                "units": "Degree true",
            },
            {
                "description": "",
                "grib_id": 167,
                "short_name": "2t",
                "cf_var": "t2m",
                "long_name": "2m_temperature",
                "type_of_level": "heightAboveGround",
                "level": 2,
                "units": "K",
            },
            {
                "description": "",
                "grib_id": 168,
                "short_name": "2d",
                "cf_var": "d2m",
                "long_name": "2m_dewpoint_temperature",
                "type_of_level": "heightAboveGround",
                "level": 2,
                "units": "K",
            },
            {
                "description": "",
                "grib_id": 174096,
                "short_name": "2sh",
                "cf_var": "sh2",
                "long_name": "2m_specific_humidity",
                "type_of_level": "heightAboveGround",
                "level": 2,
                "units": "kg kg**-1",
            },
            {
                "description": "",
                "grib_id": 134,
                "short_name": "sp",
                "cf_var": "sp",
                "long_name": "surface_pressure",
                "type_of_level": "surface",
                "level": 0,
                "units": "Pa",
            },
            {
                "description": "",
                "grib_id": 3020,
                "short_name": "vis",
                "cf_var": "vis",
                "long_name": "visibility",
                "type_of_level": "surface",
                "level": 0,
                "units": "m",
            },
            {
                "description": "",
                "grib_id": 228002,
                "short_name": "orog",
                "cf_var": "orog",
                "long_name": "orography",
                "type_of_level": "surface",
                "level": 0,
                "units": "m",
            },
        ],
        crs={
            "geographic_crs_name": "NDFD CONUS 2.5km Lambert Conformal Conic",
            "projected_crs_name": "NDFD",
            "grid_mapping_name": "lambert_conformal_conic",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31424518,
            "inverse_flattening": 298.25722356301,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "WGS84",
            "latitude_of_projection_origin": 20.191999,
            "longitude_of_projection_origin": 238.445999,
            "standard_parallel": 25,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "units": "m",
        },
    )
    era5 = DatasetConfig(
        "era5",
        "",
        "",
        "",
        datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 6, 4, tzinfo=datetime.timezone.utc),
        [],
        channels=[
            {
                "long_name": "blah",
                "cf_var": "unknown",
            }
        ],
        crs={
            "reference_ellipsoid_name": "WGS 84",
        },
    )

    api = APIConfig("mesoscaler", "mesoscaler", [urma, era5])
    api.to_disk("api.toml")


if __name__ == "__main__":
    main()
