from __future__ import annotations

import numpy as np

import src.mesoscaler as ms


class MyData(ms.DependentVariables):
    u_component_of_wind = ms.auto_field("u")
    v_component_of_wind = ms.auto_field("v")


def test_create_dataset() -> None:
    ds = ms.create.dataset(
        {
            MyData.u_component_of_wind: np.random.random((2, 2, 2)),
            MyData.v_component_of_wind: np.random.random((2, 2, 2)),
        },
        time=["2022-01-01", "2022-01-01"],
        lon=[1, 2],
        lat=[1, 2],
    )
    assert ds.dims == {"Y": 2, "X": 2, "T": 2, "Z": 1}
    assert ds.coords[ms.time].dims == ms.time.axis == ("T",) == (ms.T,)
    assert ds.coords[ms.vertical].dims == ms.vertical.axis == ("Z",) == (ms.Z,)
    assert ds.coords[ms.longitude].dims == ms.longitude.axis == ("Y", "X") == (ms.Y, ms.X)
    assert ds.coords[ms.latitude].dims == ms.latitude.axis == ("Y", "X") == (ms.Y, ms.X)
    for dvar in MyData:
        assert ds[dvar].dims == ("T", "Z", "Y", "X") == (ms.T, ms.Z, ms.Y, ms.X)
