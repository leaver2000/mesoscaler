from __future__ import annotations

import numpy as np
import pytest

import src.mesoscaler as ms


@pytest.fixture
def scale() -> ms.Mesoscale:
    return ms.Mesoscale(levels=[1013.25, 200])


def test_vertical_intersection(dataset_sequence: ms.DatasetSequence, scale: ms.Mesoscale) -> None:
    assert dataset_sequence.size == 2
    domain = scale.get_domain(dataset_sequence)
    assert np.all(domain.levels == [1013.25, 200])


def test_temporal_intersection(dataset_sequence: ms.DatasetSequence, scale: ms.Mesoscale) -> None:
    assert dataset_sequence.size == 2
    domain = scale.get_domain(dataset_sequence)
    assert np.all(domain.times == np.sort(np.unique(domain.times)))
    all_times = dataset_sequence.get_time()
    # - time_array
    time_array = np.concatenate(all_times)
    assert np.all(time_array == np.concatenate([ds.time for ds in dataset_sequence]))
    assert time_array.ndim == 1

    # - time_union
    time_union = ms.utils.nd_union(all_times, sort=True)
    assert time_union.ndim == 1
    assert time_union.size != time_array.size
    assert len(dataset_sequence) == 2
    dataset_sequence = domain.fit(dataset_sequence)
    assert len(dataset_sequence) == 2
