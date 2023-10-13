from __future__ import annotations
import sys
import datetime
import os
import abc
from typing import Iterable, Any, Generic, TypeVar, Callable
import argparse
import functools

import tqdm


class DummyClient:
    __getattr__: Callable[..., Any]

    def __init__(self, *args, **kwargs):
        raise ImportError("This client is not installed")


try:
    from s3fs import S3FileSystem as S3FSClient
except ImportError:
    S3FSClient = DummyClient
try:
    from cdsapi import Client as CDSClient
except ImportError:
    CDSClient = DummyClient


Client = TypeVar("Client")
Request = TypeVar("Request", bound=Any)
DictStrAny = dict[str, Any]


START_DATE = datetime.datetime(2019, 1, 1)
END_DATE = datetime.datetime(2021, 1, 1)


def daterange(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    timedelta: datetime.timedelta = datetime.timedelta(days=1),
) -> Iterable[datetime.datetime]:
    """
    Generate a range of dates between start_date and end_date with a given timedelta.

    Args:
        start_date (datetime.datetime): The start date of the range.
        end_date (datetime.datetime): The end date of the range.
        timedelta (datetime.timedelta, optional): The time difference between each date in the range. Defaults to datetime.timedelta(days=1).

    Yields:
        datetime.datetime: The next date in the range.

    """
    date = start_date
    while date < end_date:
        yield date
        date += timedelta


class GribGetter(Generic[Client, Request], abc.ABC):
    """Abstract base class for a generic GRIB file getter.

    This class defines the basic interface for a GRIB file getter, which is responsible for downloading
    GRIB files from a remote source and saving them to a local destination. Subclasses must implement
    the abstract methods `get_sources`, `get_filename`, and `get`.

    Attributes:
        description (str): A short description of the getter.
        source (str): The URL of the remote source.
        destination (str): The path of the local destination directory.
        client (Client): The client object used to make HTTP requests.
    """

    def __init__(
        self,
        description: str,
        source: str,
        destination: str,
        *,
        client: functools.partial[Client],
    ) -> None:
        if client is None:
            raise ImportError(f"the required dependency for {self.__class__.__name__} is not installed")
        self.description = description
        self.source = source
        if not os.path.exists(destination):
            os.makedirs(destination)
        self.destination = destination
        self.client = client()

    @abc.abstractmethod
    def get_sources(self) -> list[Request]:
        ...

    @abc.abstractmethod
    def get_filename(self, request: Request) -> str:
        ...

    @abc.abstractmethod
    def get(self, request: Request, destination: str) -> None:
        ...

    def iter(self) -> Iterable[tuple[Request, str]]:
        sources = self.get_sources()
        for source in tqdm.tqdm(sources, desc=self.description, unit="day", total=len(sources)):
            dest = self.get_filename(source)
            if not os.path.isabs(dest):
                dest = os.path.join(self.destination, dest)
            if os.path.exists(dest):
                continue
            yield source, dest


URMA2P5_DATE_FMT = "noaa-urma-pds/urma2p5.%Y%m%d"


class URMAGetter(GribGetter[S3FSClient, str]):
    def map_dates(self) -> Iterable[tuple[str, datetime.datetime]]:
        return ((x, datetime.datetime.strptime(x, URMA2P5_DATE_FMT)) for x in self.glob(self.source))

    def glob(self, request: str) -> Iterable[str]:
        return (x for x in self.client.glob(request) if isinstance(x, str))

    def get_sources(self) -> list[str]:
        return [folder for folder, date in self.map_dates() if date > START_DATE and date < END_DATE]

    def get_filename(self, request: str) -> str:
        return request.split(".")[-1]

    def get(self, request: str, destination: str) -> None:
        files = self.glob(f"s3://{request}/urma2p5.t*2dvaranl_ndfd.grb2_wexp")
        self.client.get(files, destination)


class ERA5Getter(GribGetter[CDSClient, DictStrAny]):
    GEOPOTENTIAL = "geopotential"
    SPECIFIC_HUMIDITY = "specific_humidity"
    TEMPERATURE = "temperature"
    U_COMPONENT_OF_WIND = "u_component_of_wind"
    V_COMPONENT_OF_WIND = "v_component_of_wind"
    VERTICAL_VELOCITY = "vertical_velocity"

    default_request = {
        "product_type": "reanalysis",
        "format": "grib",
        "area": [90, -180, -90, 180],
        "time": [f"{i:02}:00" for i in range(24)],
        "pressure_level": ["850", "700", "500", "300"],
        "variable": [
            GEOPOTENTIAL,
            SPECIFIC_HUMIDITY,
            TEMPERATURE,
            U_COMPONENT_OF_WIND,
            V_COMPONENT_OF_WIND,
            VERTICAL_VELOCITY,
        ],
    }

    def get_filename(self, request: DictStrAny) -> str:
        return f"{request['year']}{request['month']}{request['day']}.grib"

    def get_sources(self) -> list[dict[str, Any]]:
        return [
            self.default_request | {"year": f"{date.year:04}", "month": f"{date.month:02}", "day": f"{date.day:02}"}
            for date in daterange(START_DATE, END_DATE)
        ]

    def get(self, request: DictStrAny, destination: str) -> None:
        self.client.retrieve("reanalysis-era5-pressure-levels", request, destination)


PARTIAL_GETTERS = {
    "urma": (
        functools.partial(
            URMAGetter,
            "URMA 2.5km",
            "s3://noaa-urma-pds/urma2p5.*",
            "/mnt/data/urma2p5",
            client=functools.partial(S3FSClient, anon=True),
        ),
    ),
    "era5": (
        functools.partial(
            ERA5Getter,
            "ERA5",
            None,  # type: ignore
            "/mnt/data/era530km",
            client=functools.partial(
                CDSClient,
            ),
        ),
    ),
}  # type: dict[str, functools.partial[GribGetter[Any, Any]]]


def get_getter(resource: str, **kwargs: Any) -> GribGetter[Any, Any]:
    getter = PARTIAL_GETTERS[resource]
    return getter(**kwargs)


def main(resource: str, **kwargs: Any) -> None:
    getter = get_getter(resource, **kwargs)
    for source, destination in getter.iter():
        getter.get(source, destination)
        tqdm.tqdm.write(f"INFO: {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resource", type=str, choices=PARTIAL_GETTERS.keys())
    args = parser.parse_args()

    sys.exit(main(**vars(args)))
