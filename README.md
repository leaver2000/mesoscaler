# Mesoscaler

## Strict xarray dataset coordinate and dimension conventions

```python
>>> import mesoscaler as ms
>>> ms.Dimensions
Dimensions:
- T: T
- Z: Z
- Y: Y
- X: X
>>> ms.Coordinates
Coordinates:
-      time: time
-  vertical: vertical
-  latitude: latitude
- longitude: longitude
>>> ms.Coordinates.loc[['longitude', 'latitude']]
[longitude, latitude]
>>> ms.Coordinates.longitude.axis
>>> class MyData(ms.DependentVariables):
...     u_component_of_wind = 'u'
... 
>>> MyData
MyData:
- u_component_of_wind: u
>>> MyData('u')
u
>>> MyData('u_component_of_wind')
u
>>> ms.create_dataset(
...     {MyData('u'): np.random.random((2, 2, 2))}, lon=[1, 2], lat=[1, 2], time=["2022-01-01", "2022-01-01"]
... ) # even tho the data contained no vertical dimension it was added and assumed to be either near surface or derived atmospheric parameter
<xarray.DependentDataset>
Dimensions:    (Y: 2, X: 2, T: 2, Z: 1)
Coordinates:
    longitude  (Y, X) float64 1.0 2.0 1.0 2.0
    latitude   (Y, X) float64 1.0 1.0 2.0 2.0
    time       (T) datetime64[ns] 2022-01-01 2022-01-01
    vertical   (Z) float64 1.013e+03
Dimensions without coordinates: Y, X, T, Z
Data variables:
    u          (T, Z, Y, X) float64 0.05024 0.3504 0.2519 ... 0.8104 0.6795
Attributes:
    depends:  Dependencies(MyData)
```

## Vertical extent scaling

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
