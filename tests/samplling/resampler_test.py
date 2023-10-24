import random

import numpy as np
import pytest

import src.mesoscaler as ms


class DS1Vars(ms.DependentVariables):
    a = ms.auto_field()
    b = ms.auto_field()


class DS2Vars(ms.DependentVariables):
    a = ms.auto_field()
    b = ms.auto_field()
    c = ms.auto_field()
    d = ms.auto_field()


def create_data(
    dvars: type[ms.DependentVariables],
    n: float,
    z: list[int] = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100],
    t: int = 4,
    x: int = 256,
    y: int = 256,
):
    xx, yy = np.meshgrid(np.linspace(0, 360, x), np.linspace(0, 180, y))
    start = ms.hours.datetime("2021-01-01")
    time = ms.hours.arange(start, start + ms.hours.delta(t))

    return (
        {dvar: (np.zeros((len(time), len(z), y, x)) + i) * n for i, dvar in enumerate(dvars, 1)},
        {ms.vertical: z, ms.time: time, ms.longitude: xx, ms.latitude: yy},
    )


@pytest.mark.parametrize(
    "enum,dx,dy,height,width,time_batch_size,levels,area_of_interest",
    [
        (DS1Vars, 200, 200, 80, 80, 2, [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100], (-120, 30, -70, 25)),
        (DS1Vars, 200, 150, 80, 15, 4, [1000, 900, 800, 700, 600], (-120, 30, -70, 25)),
        (DS2Vars, 200, 30, 50, 80, 2, [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100], (-120, 30, -70, 25)),
        (DS2Vars, 100, 200, 80, 20, 4, [1000, 900, 800, 700, 600], (-120, 30, -70, 25)),
    ],
)
def test_resampler(
    enum: type[ms.DependentVariables],
    dx: int,
    dy: int,
    height: int,
    width: int,
    time_batch_size: int,
    levels: list[int],
    area_of_interest: tuple[int, ...],
) -> None:
    num = 0.25
    (ds,) = dsets = ms.create.dataset_sequence([create_data(enum, num, z=levels)])  # type: ignore

    producer = ms.create.producer(
        dsets,
        ms.AreaOfInterestSampler.partial(aoi=area_of_interest, time_batch_size=time_batch_size),
        height=height,
        width=width,
        dx=dx,
        dy=dy,
        levels=levels,
    )

    members = tuple(enum)
    for idx in random.choices(producer.indices, k=5):
        x = producer[idx]
        assert isinstance(x, np.ndarray)
        assert x.ndim == 5
        assert x.dtype == np.float_
        assert x.shape == (len(members), time_batch_size, len(levels), height, width)
        for j, member in enumerate(members, 1):
            v = num * j
            assert np.all(x[j - 1] == v)
            assert np.all(ds[member] == v)
