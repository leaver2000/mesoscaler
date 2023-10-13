import cdsapi
import glob

ALL_DAYS = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]
ALL_TIME = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
]
VARIABLES = [
    "divergence",
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]


DEFAULT_REQUEST = {
    "product_type": "reanalysis",
    "format": "grib",
    "area": [65, -140, 20, 55],
    "time": ALL_TIME,
    "day": ALL_DAYS,
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "pressure_level": "850",
}
raise Exception("This script is deprecated need to update the pressure levels" " and validate the area")

VARIABLES = [
    "divergence",
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
YEARS = ["2019", "2020", "2021"]
ROOT = "tests/data/era5"
import os


def main(out_file: str):
    year, variable, *_ = out_file.split(".")
    c = cdsapi.Client()
    assert year in YEARS
    assert variable in VARIABLES
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        DEFAULT_REQUEST | {"variable": [variable], "year": [year]},
        os.path.join(ROOT, out_file),
        # f"tests/data/{year}.{variable}.grib",
    )


if __name__ == "__main__":
    already_downloaded = [
        tuple(file.split("/")[-1].split(".")[:2]) for file in glob.glob(os.path.join(ROOT, "*.grib"))
    ]
    for year in YEARS:
        for variable in VARIABLES:
            if (year, variable) not in already_downloaded:
                print(year, variable)
                main(f"{year}.{variable}.grib")
                break
    # print(already_downloaded)
    # glob.glob(

    # )

    ...

    # main(YEARS[0], VARIABLES[0])
