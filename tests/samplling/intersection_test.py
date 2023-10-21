from __future__ import annotations

import src.mesoscaler as mesoscaler

TIME, LEVEL, LAT, LON = mesoscaler.unpack_coords()


def test_intersection(dataset_sequence: mesoscaler.DatasetSequence) -> None:
    assert dataset_sequence.size == 2
    scale = mesoscaler.Mesoscale()
    domain = dataset_sequence.get_domain(scale)
    dataset_sequence = dataset_sequence.fit(domain)
    assert dataset_sequence.size == 2
    assert dataset_sequence.batch({TIME: domain.batch_time(3)}).size == 4
