# Mesoscaler

Vertical extent scaling.

This is a python package for sampling/resampling multiple datasets for ML training loop. The datasets may have inconsistent...

- metadata conventions
- coordinate reference systems
- dimensions

This application will attempt to fit them into a single 6 dimensional tensor. PyTorch is not a required library
of this package but it does have compatibility.

The `pipeline` function is just an simple interface to the underlying `DependentDataset`, `ReSampler`, and `DataGenerator` classes.

The `DataGenerator` uses `thread` and `Queue` to generate batches of data in parallel.

```python
import functools
from torch.utils.data import DataLoader
import mesoscaler as ms

pipeline = ms.pipeline(
    [
        ("tests/data/urma.zarr", [ms.URMA.U10, ms.URMA.V10]),
        ("tests/data/era5.zarr", [ms.ERA5.U, ms.ERA5.V]),
    ],
    functools.partial(
        ms.BoundedBoxSampler,
        bbox=(-120, 30.0, -70, 25.0),
    ),
        
)


for i, batch in enumerate(DataLoader(pipeline, batch_size=12)):
    print(batch.shape)
    break
```

```text
torch.Size([12, 2, 2, 6, 80, 80])
```

TODO:

Alot...
