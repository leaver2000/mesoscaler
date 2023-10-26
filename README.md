# Mesoscaler

This code is intended for resampling multiple atmospheric datasets that have a
representation of longitude, latitude, time, and an optional Isobaric
vertical dimension. If the data does not contain a vertical dimension it is
assumed to be a derivate of the atmosphere.

## Installation

```bash
pip install .
```

## Strict xarray dataset coordinate and dimension conventions

A concrete dimension and coordinate convention is used to ensure that the
data is resampled correctly in the form of enums. The convention is as follows:

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
(Y, X)
```

```python
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

```text
torch.Size([12, 2, 2, 6, 80, 80])
```

TODO:

Alot...
