from __future__ import annotations

import src.mesoscaler as ms


def test_intersection(dataset_sequence: ms.DatasetSequence) -> None:
    assert dataset_sequence.size == 2
    scale = ms.Mesoscale()
    domain = dataset_sequence.get_domain(scale)
    dataset_sequence = dataset_sequence.fit(domain)
    assert dataset_sequence.size == 2
    assert dataset_sequence.batch({ms.time: domain.batch_time(3)}).size == 4
