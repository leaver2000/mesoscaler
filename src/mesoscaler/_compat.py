# noqa
# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
from __future__ import annotations

__all__ = [
    # - pytorch -
    "has_torch",
    "torch",
    "Dataset",
    "IterableDataset",
    "ConcatDataset",
    "ChainDataset",
    "BatchSampler",
    "Sampler",
    "SequentialSampler",
    "DataLoader",
    # - tqdm -
    "tqdm",
    # - matplotlib -
    "has_matplotlib",
    "plt",
    # - cartopy -
    "has_cartopy",
    "ccrs",
    "cfeature",
    "GeoAxes",
]
import contextlib
import typing


class _dummy:
    def __init__(self, error_message: str = "module not installed!") -> None:
        self.error_message = error_message

    def __getattr__(self, _) -> typing.NoReturn:
        raise RuntimeError(self.error_message)

    def __call__(self, *_, **__) -> typing.NoReturn:
        raise RuntimeError(self.error_message)


try:
    from tqdm import tqdm

    with contextlib.suppress(NameError):
        get_ipython  # type: ignore
        from tqdm.notebook import tqdm

except ImportError:
    tqdm = lambda x, *_, **__: iter(x)


if typing.TYPE_CHECKING:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes


try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    plt = _dummy("matplotlib not installed")


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.geoaxes import GeoAxes

    _projections = {
        "laea": ccrs.LambertAzimuthalEqualArea,
        "lcc": ccrs.LambertConformal,
        "aea": ccrs.AlbersEqualArea,
    }
    LiteralProjection = typing.Literal[
        "laea",
        "lcc",
        "aea",
    ]

    def get_projection(
        name: LiteralProjection, central_longitude: float = 0.0, central_latitude: float = 0.0
    ) -> ccrs.Projection:
        proj = _projections.get(name, None)
        if proj is None:
            raise ValueError(f"projection {name} not found!")
        return proj(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
        )

    has_cartopy = True
except ImportError:
    has_cartopy = False
    ccrs = _dummy("cartopy not installed")
    GeoAxes = _dummy("cartopy not installed")
    cfeature = _dummy("cartopy not installed")


try:
    import torch  # noqa

    has_torch = True
except ImportError:
    has_torch = False
    torch = _dummy("torch not installed")


if has_torch:
    from torch.utils.data import (
        ChainDataset,
        ConcatDataset,
        DataLoader,
        Dataset,
        IterableDataset,
    )
    from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler


elif not has_torch and not typing.TYPE_CHECKING:
    import bisect
    import warnings
    from typing import (
        Generic,
        Iterable,
        Iterator,
        List,
        Optional,
        Sized,
        TypeVar,
        Union,
    )

    T_co = TypeVar("T_co", covariant=True)
    DataLoader = None

    # =================================================================================================================
    # torch.utils.data.sampler
    # =================================================================================================================
    class Dataset(Generic[T_co]):
        def __getitem__(self, index) -> T_co:
            raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

        def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
            return ConcatDataset([self, other])

    class IterableDataset(Dataset[T_co]):
        def __iter__(self) -> Iterator[T_co]:
            raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

        def __add__(self, other: Dataset[T_co]):
            return ChainDataset([self, other])

    class ConcatDataset(Dataset[T_co]):
        datasets: List[Dataset[T_co]]
        cumulative_sizes: List[int]

        @staticmethod
        def cumsum(sequence):
            r, s = [], 0
            for e in sequence:
                l = len(e)
                r.append(l + s)
                s += l
            return r

        def __init__(self, datasets: Iterable[Dataset]) -> None:
            super().__init__()
            self.datasets = list(datasets)
            assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
            for d in self.datasets:
                assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
            self.cumulative_sizes = self.cumsum(self.datasets)

        def __len__(self):
            return self.cumulative_sizes[-1]

        def __getitem__(self, idx):
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]

        @property
        def cummulative_sizes(self):
            warnings.warn(
                "cummulative_sizes attribute is renamed to " "cumulative_sizes", DeprecationWarning, stacklevel=2
            )
            return self.cumulative_sizes

    class ChainDataset(IterableDataset):
        def __init__(self, datasets: Iterable[Dataset]) -> None:
            super().__init__()
            self.datasets = datasets

        def __iter__(self):
            for d in self.datasets:
                assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
                yield from d

        def __len__(self):
            total = 0
            for d in self.datasets:
                assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
                total += len(d)  # type: ignore[arg-type]
            return total

    # =================================================================================================================
    # torch.utils.data.sampler
    # =================================================================================================================
    class Sampler(Generic[T_co]):
        def __init__(self, data_source: Optional[Sized] = None) -> None:
            if data_source is not None:
                import warnings

                warnings.warn(
                    "`data_source` argument is not used and will be removed in 2.2.0."
                    "You may still have custom implementation that utilizes it."
                )

        def __iter__(self) -> Iterator[T_co]:
            raise NotImplementedError

    class SequentialSampler(Sampler[int]):
        data_source: Sized

        def __init__(self, data_source: Sized) -> None:
            self.data_source = data_source

        def __iter__(self) -> Iterator[int]:
            return iter(range(len(self.data_source)))

        def __len__(self) -> int:
            return len(self.data_source)

    class BatchSampler(Sampler[List[int]]):
        def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
            # Since collections.abc.Iterable does not check for `__getitem__`, which
            # is one way for an object to be an iterable, we don't do an `isinstance`
            # check here.
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
                raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
            if not isinstance(drop_last, bool):
                raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
            self.sampler = sampler
            self.batch_size = batch_size
            self.drop_last = drop_last

        def __iter__(self) -> Iterator[List[int]]:
            # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
            if self.drop_last:
                sampler_iter = iter(self.sampler)
                while True:
                    try:
                        batch = [next(sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                    except StopIteration:
                        break
            else:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for idx in self.sampler:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                if idx_in_batch > 0:
                    yield batch[:idx_in_batch]

        def __len__(self) -> int:
            # Can only be called if self.sampler has __len__ implemented
            # We cannot enforce this condition, so we turn off typechecking for the
            # implementation below.
            # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
            if self.drop_last:
                return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
            else:
                return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
