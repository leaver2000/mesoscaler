import src.mesoscaler as ms


def test_mesoscale_unit_handling() -> None:
    scale = ms.Mesoscale(200, 200, levels=[1013.25], xy_units="km")
    assert scale.to_numpy(units="km").tolist() == [[-200, -200, 200, 200]]
    assert scale.to_numpy(units="m").tolist() == [[-200_000, -200_000, 200_000, 200_000]]

    scale = ms.Mesoscale(200_000, levels=[1013.25], xy_units="m")
    assert scale.to_numpy(units="m").tolist() == [[-200_000, -200_000, 200_000, 200_000]]
    assert scale.to_numpy(units="km").tolist() == [[-200, -200, 200, 200]]
